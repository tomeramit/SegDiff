from pathlib import Path

import h5py
import os
import cv2
import numpy as np
from cv2 import resize


def get_img(cfile):
    img = cv2.cvtColor(cv2.imread(cfile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
    return img


def get_mask(cfile):
    GT = cv2.imread(cfile, 0)
    GT = resize(GT, (256, 256), interpolation=cv2.INTER_LINEAR)
    GT[GT >= 0.5] = 1
    GT[GT < 0.5] = 0
    return GT


def main(args, out_path):
    data_folder_path = Path(args['path'])
    imgs_list = sorted(list(data_folder_path.glob("building_[0-9]*.tif")))
    masks_list = sorted(list(data_folder_path.glob("building_mask_[0-9]*.tif")))

    hf_tri = h5py.File(str(out_path / "full_training_vaih.hdf5"), 'w')
    hf_test = h5py.File(str(out_path / "full_test_vaih.hdf5"), 'w')

    imgs_tri = hf_tri.create_group('imgs')
    mask_single_tri = hf_tri.create_group('mask_single')

    imgs_test = hf_test.create_group('imgs')
    mask_single_test = hf_test.create_group('mask_single')

    for image_path in imgs_list[:100]:
        print('training: ' + str(image_path))
        img = get_img(str(image_path))
        imgs_tri.create_dataset(image_path.stem, data=img, dtype=np.uint8)

    for image_path in imgs_list[100:]:
        print('validation: ' + str(image_path))
        img = get_img(str(image_path))
        imgs_test.create_dataset(image_path.stem, data=img, dtype=np.uint8)

    for mask_path in masks_list[:100]:
        print('training: ' + str(mask_path))
        mask = get_mask(str(mask_path))
        mask_single_tri.create_dataset(mask_path.stem, data=mask, dtype=np.uint8)

    for mask_path in masks_list[100:]:
        print('validation: ' + str(mask_path))
        mask = get_mask(str(mask_path))
        mask_single_test.create_dataset(mask_path.stem, data=mask, dtype=np.uint8)

    hf_tri.close()
    hf_test.close()


if __name__ == '__main__':
    import argparse
    folder_path = Path(__file__).absolute().parent.parent.parent / "data" / "Vaihingen"
    folder_path.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-path',
                        '--path',
                        default='',
                        help='Data path, should point on "building"',
                        required=True)
    args = vars(parser.parse_args())
    main(args, out_path=folder_path)



