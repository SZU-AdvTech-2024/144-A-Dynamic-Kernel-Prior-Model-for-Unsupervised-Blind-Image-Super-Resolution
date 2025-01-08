# A Dynamic Kernel Prior Model for Unsupervised Blind Image Super-Resolution (DKP), CVPR2024

This repository is the official PyTorch implementation of DKP to Blind Super-Resolution 
([arXiv](https://arxiv.org/abs/2404.15620), [supp](https://github.com/XYLGroup/DKP)).

## Requirements

- pip install numpy torch blobfile tqdm pyYaml pillow    # e.g. torch 1.7.1+cu110.

### Pre-Trained Models for DiffDKP

To restore general images, download this [model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)(from [guided-diffusion](https://github.com/openai/guided-diffusion)) and put it into `DiffDKP/data/pretrained/`.

```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
```

Note that the pre-trained models are only used for DiffDKP, DIPDKP is processed without pre-trained models.

## Quick Run DIPDKP

To run the code without preparing data, run this command:

```bash
cd DIPDKP/DIPDKP
python main.py
```

## Quick Run DiffDKP

To run the code without preparing data, run this command:

```bash
cd DiffDKP
python main.py
```

---

## Data Preparation for DIPDKP

To prepare testing data, please organize images as `data/datasets/Set5/HR/baby.png`, and run this command:

```bash
cd DIPDKP/data
python prepare_dataset.py --model DIPDKP --sf 2 --dataset Set5
```

## Data Preparation for DiffDKP

To prepare testing data, please organize images as `data/datasets/deblur/Set5/HR_256/baby.png`, and run this command:

```bash
cd DiffDKP/data
python prepare_dataset.py --model DIPDKP --sf 2 --dataset Set5
```

Commonly used datasets can be downloaded [here](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets).



# 
