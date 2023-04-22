from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mpi4py import MPI
from torch.utils.data import Dataset, DataLoader

from datasets.transforms import \
    Compose, ToPILImage, Resize, RandomHorizontalFlip, ToTensor, Normalize, \
    RandomAffine, RandomVerticalFlip, ColorJitter


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
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

    dataset = VaihDataset(
        mode='train',
        image_size=image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
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


class VaihDataset(Dataset):

    CLASSES = ('building',)

    PALETTE = [[255, 0, 0]]

    def __init__(self, mode, std=np.array([0.22645572 * 255, 0.15276193 * 255, 0.140702 * 255]),
                 mean=np.array([0.47341759 * 255, 0.28791303 * 255, 0.2850705 * 255]), no_aug=False,
                 image_size=256, max_data_size=None, shard=0, num_shards=1, small_image_size=None):

        self.mode = mode
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

        if mode == 'train' and not no_aug:
            self.transformations = Compose([ToPILImage(),
                                                        Resize(size=(image_size, image_size)),
                                                        RandomAffine(degrees=[0, 360], scale=(0.75, 1.5)),
                                                       ColorJitter(brightness=0.6,
                                                                              contrast=0.5,
                                                                              saturation=0.4,
                                                                              hue=0.025),
                                                       RandomVerticalFlip(),
                                                       RandomHorizontalFlip(),
                                                       ToTensor(),
                                                       Normalize(self.mean, self.std)])
        else:
            self.transformations = Compose([ToPILImage(),
                                            Resize(size=(image_size, image_size)),
                                                       ToTensor(),
                                                       Normalize(self.mean, self.std)])
        if mode == 'train':
            self.data_length = 100
        else:
            self.data_length = 68

        if max_data_size is not None:
            self.data_length = max_data_size

        if self.mode == 'train':
            self.data = h5py.File(
                str(Path(__file__).absolute().parent.parent.parent / "data/Vaihingen/full_training_vaih.hdf5"), 'r')

        else:
            self.data = h5py.File(
                str(Path(__file__).absolute().parent.parent.parent / "data/Vaihingen/full_test_vaih.hdf5"), 'r')

        self.small_image_size = small_image_size
        self.mask = self.data['mask_single']
        self.imgs = self.data['imgs']
        self.img_list = list(self.imgs)[shard::num_shards]
        self.mask_list = list(self.mask)[shard::num_shards]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        cimage = self.img_list[item]
        img = np.array(self.imgs.get(cimage))
        cmask = self.mask_list[item]
        mask = np.array(self.mask.get(cmask))
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        img, mask = self.transformations(img, mask)
        out_dict = {"conditioned_image": img}
        mask = (2 * mask - 1.0).unsqueeze(0)
        if self.small_image_size is not None:
            out_dict["low_res"] = F.interpolate(mask.unsqueeze(0), self.small_image_size, mode="nearest").squeeze(0)
        return mask, out_dict, str(Path(cimage).stem)


if __name__ == '__main__':
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    dataset = VaihDataset('train', mean=mean, std=std, image_size=256)
    dataset2 = VaihDataset('train', mean=mean, std=std, image_size=256, no_aug=True)
    for i in range(10):
        mask, out_dict, _ = dataset[0]
        img = out_dict["conditioned_image"]
        plt.imshow(img.permute(1,2,0).numpy().astype(np.uint8))
        plt.show()

        plt.imshow(mask.permute(1,2,0).numpy(), cmap='gray')
        plt.show()

        mask, out_dict, _ = dataset2[0]
        img = out_dict["conditioned_image"]
        plt.imshow(img.permute(1,2,0).numpy().astype(np.uint8))
        plt.show()