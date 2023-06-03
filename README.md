This is the official repository of the paper [SegDiff: Image Segmentation with Diffusion Probabilistic Models](https://arxiv.org/abs/2112.00390)

The code is based on [Improved Denoising Diffusion Probabilistic Models.](https://github.com/openai/improved-diffusion)

## Installation
### Conda environment
To create the environment use the conda environment command
```
conda env create -f environment.yml
```

## Project structure and data preparations
our project need to be arranged in the following format

```
segdiff/ # git clone the source code here

data/ # the root of the data folders
    Vaihingen/
    Medical/MoNuSeg/
    cityscapes_instances/
```

### Vaihingen

download the dataset from [link](https://drive.google.com/file/d/1nenpWH4BdplSiHdfXs0oYfiA5qL42plB/view) 
and unzip it's content (folder named buildings), execute the preprocess
```
datasets/preprocess_vaihingen.py --path building-folder-path 
```

Vaihingen dataset should have the following format
```
Vaihingen/
    full_test_vaih.hdf5
    full_training_vaih.hdf5
```

### MonuSeg
general [website](https://monuseg.grand-challenge.org/) of the challenge,
download the dataset
[train](https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view?usp=sharing)
and [test](https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view?usp=sharing) sets.

launch the matlab [code](https://drive.google.com/file/d/1YDtIiLZX0lQzZp_JbqneHXHvRo45ZWGX/view) 
for preprocess 

MonuSeg dataset should have the following format
```
MonuSeg/
    Test/
        img/
            XX.tif
        mask/
            XX.png
    Training/
        img/
            XX.tif
        mask/
            XX.png
```

### Cityscapes

download [cityscapes](https://www.cityscapes-dataset.com) dataset with the splits from 
[PolyRNN++](https://github.com/fidler-lab/polyrnn-pp), follow the instructions [here](https://github.com/shirgur/ACDRNet) for preparations

To get cityscapes_final_v5 annotations you can sign up to get PolygonRNN++ code here http://www.cs.toronto.edu/polyrnn/code_signup/ the cityscapes_final_v5 folder is inside the data folder

Cityscapes dataset should have the following format
```
cityscapes_instances/
    full/
        all_classes_instances.json
    train/
        all_classes_instances.json
    train_val/
        all_classes_instances.json
    val/
        all_classes_instances.json
    all_images.hdf5
```


## Train and Evaluate
Execute the following commands (multi gpu is supported for training, set the gpus with CUDA_VISIBLE_DEVICES and -n for the actual number)

Training options:
```
# Training
--batch-size    Batch size
--lr            Learning rate

# Architecture
--rrdb_blocks       Number of rrdb blocks
--dropout           Dropout
--diffusion_steps   number of steps for the diffusion model

# Cityscapes
--class_name        name of class of cityscapes, options are ["bike", "bus", "person", "train", "motorcycle", "car", "rider"]
--expansion         boolean flag, for expansion setting or not

# Misc
--save_interval     interval for saving model weights
```

### MonuSeg
Training script example:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 image_train_diff_medical.py --rrdb_blocks 12 --batch_size 2 --lr 0.0001 --diffusion_steps 100
```

Evaluation script example:
```
CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python image_sample_diff_medical.py --model_path path-for-model-weights
```

### Cityscapes
Training script example:
```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python image_train_diff_city.py --class_name "train" --expansion True --rrdb_blocks 15 --lr 0.0001 --batch_size 15 --diffusion_steps 100
```

Evaluation script example:
```
CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python image_sample_diff_city.py --model_path path-for-model-weights
```

### Vaihingen
Training script example:
```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python image_train_diff_vaih.py --lr 0.0001 --batch_size 4 --dropout 0.1 --rrdb_blocks 6 --diffusion_steps 100
```

Evaluation script example:
```
CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python image_sample_diff_vaih.py --model_path path-for-model-weights
```

## Citation
```
@article{amit2021segdiff,
  title={Segdiff: Image segmentation with diffusion probabilistic models},
  author={Amit, Tomer and Nachmani, Eliya and Shaharbany, Tal and Wolf, Lior},
  journal={arXiv preprint arXiv:2112.00390},
  year={2021}
}
```

