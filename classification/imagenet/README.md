# CARD: Classification and Regression Diffusion Models - ImageNet experiments

This repo contains the official implementation for the paper [CARD: Classification and Regression Diffusion Models](https://arxiv.org/pdf/2206.07275.pdf) by [Xizewen Han](https://www.linkedin.com/in/xizewenhan/), [Huangjie Zheng](https://huangjiezheng.com/), and [Mingyuan Zhou](https://mingyuanzhou.github.io/). Published in NeurIPS 2022 (poster). This folder provide instruction for running large-scale experiments, e.g., ImageNet experiments with CARD.

--------------------

## How to Run the Code

### Usage

**Preparation**
Before starting running the code, please first download the checkpoint weight of $\boldsymbol{f_{\phi}}$ (ResNet50 here) from [here](https://download.pytorch.org/models/resnet50-11ad3fa6.pth).

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

**Running code**

```python
python main.py \
        -a resnet50 --epochs 200 \
        --multiprocessing-distributed --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
        --world-size $WORLD_SIZE --rank $RANK \
        --pretrained resnet50-11ad3fa6.pth \
        --num_classes 1000 --lars \
        /path/to/imagenet 
```
This command enables training with multiple GPUs and multiple nodes. For single node (single/multi-GPU), please set MASTER_ADDR=localhost, MASTER_PORT as any preferred number, WORLD_SIZE=1 and RANK=0. For multi-node training, please refer to your system configuration, and set these os parameters accordingly.


### Acknowledgement

This repo is built and modified from the repo of [MoCo](https://github.com/facebookresearch/moco) and [Simsiam](https://github.com/facebookresearch/simsiam). We appreciate their great work.

## References

If you find the code helpful for your research, please consider citing
```bib
@inproceedings{han2022card,
  title={CARD: Classification and Regression Diffusion Models},
  author={Han, Xizewen and Zheng, Huangjie and Zhou, Mingyuan},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022}
}
```
