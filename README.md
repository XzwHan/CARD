# CARD: Classification and Regression Diffusion Models

This repo contains the official implementation for the paper [CARD: Classification and Regression Diffusion Models](https://arxiv.org/pdf/2206.07275.pdf) by [Xizewen Han](https://www.linkedin.com/in/xizewenhan/), [Huangjie Zheng](https://huangjiezheng.com/), and [Mingyuan Zhou](https://mingyuanzhou.github.io/). Published in NeurIPS 2022 (poster).

--------------------

## How to Run the Code

### Dependencies

We recommend configuring the environment through [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html). Run the following to install and to activate the environment 

```sh
conda env create -f environment.yml
conda activate card
```

The name of the environment is set to **card** by default. You can modify the first line of the `environment.yml` file to set the new environment's name.

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
