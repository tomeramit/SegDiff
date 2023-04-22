import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image
from matplotlib import pyplot as plt
from mpi4py import MPI
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm

from datasets.transforms import \
    Compose, ToPILImage, RandomHorizontalFlip, ToTensor, Normalize, RandomAffine


def create_dataset(mode="train", class_name="train", expansion=False):
    shard=MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()
    data_inst_path = str(Path(__file__).absolute().parent.parent.parent / "data/cityscapes_instances/")

    print('loading \"{}\" annotations into memory...'.format(mode))
    data = json.load(open(os.path.join(data_inst_path, mode, 'all_classes_instances.json'), 'r'))

    annotations = data['data'][class_name][shard::num_shards]

    hdf5_obj = h5py.File(os.path.join(data_inst_path, 'all_images.hdf5'), 'r')
    images = [hdf5_obj[ann['img']['file_name']] for ann in annotations]

    return CityscapesInstances(
        images,
        annotations,
        mode=mode,
        expansion=expansion
    )


def load_data(
    *, data_dir, batch_size, image_size, class_name, class_cond=False, expansion, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    dataset = create_dataset(mode="train", class_name=class_name, expansion=expansion)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


class CityscapesInstances(Dataset):
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def __init__(self,
                 images,
                 annotations,
                 no_aug=False,
                 mode='train',
                 loops=100,
                 expansion=False,
                 std=np.array([58.395, 57.12, 57.375]),
                 mean=np.array([123.675, 116.28, 103.53]),
                 ):
        super(CityscapesInstances, self).__init__()

        self.loops = loops
        self.mode = mode
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.expansion = expansion
        image_size = 128

        if mode == 'train' and not no_aug:
            self.transformations = Compose([
                ToPILImage(),
                # Resize((image_size, image_size)),
                RandomHorizontalFlip(),
                RandomAffine(22, scale=(0.75, 1.25)),
                ToTensor(),
                Normalize(self.mean, self.std)
                # transforms.NormalizeInstance()
            ])
        else:
            self.transformations = Compose([
                ToPILImage(),
                # Resize((image_size, image_size), do_mask=False),
                ToTensor(),
                Normalize(self.mean, self.std),
                # transforms.NormalizeInstance()
            ])

        self.instance_images = []
        self.instance_masks = []

        self.annotations = annotations

        for item in tqdm(range(len(images))):
            ann = self.annotations[item]
            mask = self._poly2mask(ann['segmentation'], ann['img']['height'], ann['img']['width'])
            bbox = np.maximum(0, np.array(ann['bbox']).astype(np.int32))

            if self.expansion:
                if self.mode == 'train':
                    bounding_box_expansion = random.randint(10, 20)
                else:
                    bounding_box_expansion = 15

                increase_axis_by = bbox[3] * (bounding_box_expansion / 100)
                increase_each_coordinate = increase_axis_by / 2

                x_1 = bbox[1] - increase_each_coordinate
                x_2 = bbox[1] + bbox[3] + increase_each_coordinate

                increase_axis_by = bbox[2] * (bounding_box_expansion / 100)
                increase_each_coordinate = increase_axis_by / 2

                y_1 = bbox[0] - increase_each_coordinate
                y_2 = bbox[0] + bbox[2] + increase_each_coordinate

                # check the axis order
                x_2 = round(min(x_2, images[item].shape[0]))
                y_2 = round(min(y_2, images[item].shape[1]))

                x_1 = round(max(x_1, 0))
                y_1 = round(max(y_1, 0))

                instance_image = images[item][x_1:x_2, y_1:y_2]
                instance_mask = mask[x_1:x_2, y_1:y_2]
            else:
                instance_image = images[item][bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                instance_mask = mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

            size = [image_size, image_size]
            self.instance_images.append(resize(torch.from_numpy(instance_image).permute(2, 0, 1), size, Image.BILINEAR).permute(1, 2, 0).numpy())

            if mode == 'train' and not no_aug:
                self.instance_masks.append(resize(torch.from_numpy(instance_mask).unsqueeze(0), size, Image.NEAREST).squeeze(0).numpy())
            else:
                self.instance_masks.append(instance_mask)

    @staticmethod
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        ann = self.annotations[item]

        instance_image, instance_mask = self.transformations(self.instance_images[item], self.instance_masks[item])

        out_dict = {"conditioned_image": instance_image}
        instance_mask = 2 * instance_mask - 1.0
        return instance_mask.unsqueeze(0), out_dict, Path(ann["img"]['file_name']).stem


def main():
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    dataset = create_dataset(class_name="train", mode='train')
    for i in range(10):
        # mask, out_dict, _ = dataset[i]
        # img = out_dict["conditioned_image"]
        # plt.imshow(img.permute(1, 2, 0).numpy().astype(np.uint8))
        # plt.show()
        #
        # plt.imshow(mask.permute(1, 2, 0).numpy(), cmap='gray')
        # plt.show()

        masks, out_dict, _ = dataset[i]
        imgs = out_dict["conditioned_image"]
        for index in range(10):
            plt.imshow(imgs[index * 10].permute(1, 2, 0).numpy().astype(np.uint8))
            plt.show()

        for index in range(10):
            plt.imshow(masks[index * 10].permute(1, 2, 0).numpy(), cmap='gray')
            plt.show()

        pass


if __name__ == '__main__':
    main()

