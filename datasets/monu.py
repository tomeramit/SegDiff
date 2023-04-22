import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from mpi4py import MPI
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.transforms import \
    Compose, ToPILImage, ColorJitter, RandomHorizontalFlip, ToTensor, Normalize, RandomVerticalFlip, RandomAffine, \
    Resize, RandomCrop


def cv2_loader(path, is_mask):
    if is_mask:
        # img = cv2.imread(path, 0)
        img = imageio.imread(path)
        img[img > 0] = 1
    else:
        # img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # img = imageio.imread(path)
        img = tifffile.imread(path)
    return img


def get_monu_transform(image_size):

    transform_train = Compose([
        ToPILImage(),
        Resize((512, 512)),
        RandomCrop((image_size, image_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(int(22), scale=(float(0.75), float(1.25))),
        ColorJitter(brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1),
        ToTensor(),
        Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    return transform_train, transform_test


def create_dataset(mode="train", image_size=256):
    datadir = str(Path(__file__).absolute().parent.parent.parent / "data/Medical/MoNuSeg")

    transform_train, transform_test = get_monu_transform(image_size)
    if mode == "train":
        return MonuDataset(datadir, train=True, transform=transform_train, image_size=image_size)
    else:
        return MonuDataset(datadir, train=False, transform=transform_test)


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

    dataset = create_dataset(mode="train")

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


class MonuDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=cv2_loader, pSize=8, image_size=256):
        self.root = root
        if train:
            self.imgs_root = os.path.join(self.root, 'Training', 'img')
            self.masks_root = os.path.join(self.root, 'Training', 'mask')
        else:
            self.imgs_root = os.path.join(self.root, 'Test', 'img')
            self.masks_root = os.path.join(self.root, 'Test', 'mask')
        self.image_size = image_size
        self.paths = sorted(os.listdir(self.imgs_root))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.pSize = pSize
        self.masks = []
        self.imgs = []
        self.mean = torch.from_numpy(np.array([142.07, 98.48, 132.96]))
        self.std = torch.from_numpy(np.array([65.78, 57.05, 57.78]))

        shard = MPI.COMM_WORLD.Get_rank()
        num_shards = MPI.COMM_WORLD.Get_size()

        for file_path in tqdm(self.paths):
            mask_path = file_path.split('.')[0] + '.png'
            self.imgs.append(self.loader(os.path.join(self.imgs_root, file_path), is_mask=False))
            self.masks.append(self.loader(os.path.join(self.masks_root, mask_path), is_mask=True))

        self.imgs = self.imgs[shard::num_shards]
        self.masks = self.masks[shard::num_shards]
        self.paths = self.paths[shard::num_shards]

        print('num of data:{}'.format(len(self.paths)))

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]

        img, mask = self.transform(img, mask)
        out_dict = {"conditioned_image": img}
        mask = 2 * mask - 1.0
        return mask.unsqueeze(0), out_dict, f"{Path(self.paths[index]).stem}_{index}"

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    val_dataset = create_dataset(
        mode='val',
        image_size=256,
    )

    ds = torch.utils.data.DataLoader(val_dataset,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=True)
    pbar = tqdm(ds)
    mean0_list = []
    mean1_list = []
    mean2_list = []
    std0_list = []
    std1_list = []
    std2_list = []
    for i, (mask, out_dict, _) in enumerate(pbar):
        img = out_dict["conditioned_image"]
        plt.imshow(img.squeeze().permute(1,2,0).numpy().astype(np.uint8))
        plt.show()

        plt.imshow(mask.squeeze().numpy(), cmap='gray')
        plt.show()
        a = img.mean(dim=(0, 2, 3))
        b = img.std(dim=(0, 2, 3))
        mean0_list.append(a[0].item())
        mean1_list.append(a[1].item())
        mean2_list.append(a[2].item())
        std0_list.append(b[0].item())
        std1_list.append(b[1].item())
        std2_list.append(b[2].item())
    print(np.mean(mean0_list))
    print(np.mean(mean1_list))
    print(np.mean(mean2_list))

    print(np.mean(std0_list))
    print(np.mean(std1_list))
    print(np.mean(std2_list))

        # a = img.squeeze().permute(1, 2, 0).cpu().numpy()
        # b = mask.squeeze().cpu().numpy()
        # a = (a - a.min()) / (a.max() - a.min())
        # cv2.imwrite('kaki.jpg', 255*a)
        # cv2.imwrite('kaki_mask.jpg', 255*b)