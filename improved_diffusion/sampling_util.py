import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
from PIL import Image
from kornia import denormalize
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import dist_util
from .metrics import FBound_metric, WCov_metric
from datasets.monu import MonuDataset
from .utils import set_random_seed_for_iterations

cityspallete = [
    0, 0, 0,
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten()), \
           WCov_metric(predict, target), FBound_metric(predict, target)


def sampling_major_vote_func(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, step, n_rounds=3):
    ddp_model.eval()
    batch_size = 1
    major_vote_number = 9
    loader = DataLoader(dataset, batch_size=batch_size)
    loader_iter = iter(loader)

    f1_score_list = []
    miou_list = []
    fbound_list = []
    wcov_list = []

    with torch.no_grad():
        for round_index in tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
        ):
            gt_mask, condition_on, name = next(loader_iter)
            set_random_seed_for_iterations(step + int(name[0].split("_")[1]))
            gt_mask = (gt_mask + 1.0) / 2.0
            condition_on = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())

            for i in range(gt_mask.shape[0]):
                gt_img = Image.fromarray(gt_mask[i][0].detach().cpu().numpy().astype('uint8'))
                gt_img.putpalette(cityspallete)
                gt_img.save(
                    os.path.join(output_folder, f"{name[i]}_gt_palette.png"))
                gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() - 1).astype(np.uint8))
                gt_img.save(
                    os.path.join(output_folder, f"{name[i]}_gt.png"))

            for i in range(condition_on.shape[0]):
                denorm_condition_on = denormalize(condition_on.clone(), mean=dataset.mean, std=dataset.std)
                tvu.save_image(
                    denorm_condition_on[i,] / 255.,
                    os.path.join(output_folder, f"{name[i]}_condition_on.png")
                )

            if isinstance(dataset, MonuDataset):
                _, _, W, H = former_frame_for_feature_extraction.shape
                kernel_size = dataset.image_size
                stride = 256
                patches = []
                for y, x in np.ndindex((((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1)):
                    y = y * stride
                    x = x * stride
                    patches.append(former_frame_for_feature_extraction[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)])
                patches = torch.stack(patches)

                major_vote_list = []
                for i in range(major_vote_number):
                    x_list = []

                    for index in range(math.ceil(patches.shape[0] / 4)):
                        model_kwargs = {"conditioned_image": patches[index * 4: min((index + 1) * 4, patches.shape[0])]}
                        x = diffusion_model.p_sample_loop(
                                ddp_model,
                                (model_kwargs["conditioned_image"].shape[0], gt_mask.shape[1], model_kwargs["conditioned_image"].shape[2], model_kwargs["conditioned_image"].shape[3]),
                                progress=True,
                                clip_denoised=clip_denoised,
                                model_kwargs=model_kwargs
                            )

                        x_list.append(x)
                    out = torch.cat(x_list)

                    output = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    idx_sum = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1], former_frame_for_feature_extraction.shape[2], former_frame_for_feature_extraction.shape[3]))
                    for index, val in enumerate(out):
                        y, x = np.unravel_index(index, (((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1))
                        y = y * stride
                        x = x * stride

                        idx_sum[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += 1

                        output[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += val[:, :min(y + kernel_size, W) - y, :min(x + kernel_size, H) - x].cpu().data.numpy()

                    output = output / idx_sum
                    major_vote_list.append(output)

                x = torch.cat(major_vote_list)

            else:
                model_kwargs = {
                    "conditioned_image": torch.cat([former_frame_for_feature_extraction] * major_vote_number)}

                x = diffusion_model.p_sample_loop(
                    ddp_model,
                    (major_vote_number, gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                     former_frame_for_feature_extraction.shape[3]),
                    progress=True,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs
                )

            x = (x + 1.0) / 2.0

            if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
                x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            x = torch.clamp(x, 0.0, 1.0)

            # major vote result
            x = x.mean(dim=0, keepdim=True).round()

            for i in range(x.shape[0]):
                # save as outer training ids
                # current_output = x[i][0] + 1
                # current_output[current_output == current_output.max()] = 0
                out_img = Image.fromarray(x[i][0].detach().cpu().numpy().astype('uint8'))
                out_img.putpalette(cityspallete)
                out_img.save(
                    os.path.join(output_folder, f"{name[i]}_model_output_palette.png"))
                out_img = Image.fromarray((x[i][0].detach().cpu().numpy() - 1).astype(np.uint8))
                out_img.save(
                    os.path.join(output_folder, f"{name[i]}_model_output.png"))

            for index, (gt_im, out_im) in enumerate(zip(gt_mask, x)):

                f1, miou, wcov, fbound = calculate_metrics(out_im[0], gt_im[0])
                f1_score_list.append(f1)
                miou_list.append(miou)
                wcov_list.append(wcov)
                fbound_list.append(fbound)

                logger.info(
                    f"{name[index]} iou {miou_list[-1]}, f1_Score {f1_score_list[-1]}, WCov {wcov_list[-1]}, boundF {fbound_list[-1]}")

    my_length = len(miou_list)
    length_of_data = torch.tensor(len(miou_list), device=dist_util.dev())
    gathered_length_of_data = [torch.tensor(1, device=dist_util.dev()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    iou_tensor = torch.tensor(miou_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    f1_tensor = torch.tensor(f1_score_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    wcov_tensor = torch.tensor(wcov_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    boundf_tensor = torch.tensor(fbound_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    gathered_miou = [torch.ones_like(iou_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_f1 = [torch.ones_like(f1_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_wcov = [torch.ones_like(wcov_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_boundf = [torch.ones_like(boundf_tensor) * -1 for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_miou, iou_tensor)
    dist.all_gather(gathered_f1, f1_tensor)
    dist.all_gather(gathered_wcov, wcov_tensor)
    dist.all_gather(gathered_boundf, boundf_tensor)

    # if dist.get_rank() == 0:
    logger.info("measure total avg")
    gathered_miou = torch.cat(gathered_miou)
    gathered_miou = gathered_miou[gathered_miou != -1]
    logger.info(f"mean iou {gathered_miou.mean()}")

    gathered_f1 = torch.cat(gathered_f1)
    gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean f1 {gathered_f1.mean()}")
    gathered_wcov = torch.cat(gathered_wcov)
    gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean WCov {gathered_wcov.mean()}")
    gathered_boundf = torch.cat(gathered_boundf)
    gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean boundF {gathered_boundf.mean()}")

    dist.barrier()
    return gathered_miou.mean().item()
