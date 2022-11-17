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

### Usage

We organize our code by the type of tasks into the corresponding `regression` and `classification` directories. As we assume the response variable $\boldsymbol{y}$ to reside in the real continuous space for both regression and classification tasks, the code for training and inference of the diffusion models are the same (`diffusion_utils.py` file). The main differences are the handling of the $\boldsymbol{f_{\phi}}$ model (named `self.cond_pred_model` in both `card_regression.py` and `card_classification.py`), the evaluation of trained models (class function `test`), and the neural network architecture (`model.py` file).

We provide the scripts of model training and evaluation for the tasks reported in our paper in the `training_scripts` directory, including:

* Regression
  * 8 toy examples: linear regression, quadratic regression, sinusoidal regression, log-log linear regression, log-log cubic regression, inverse sinusoidal regression, 8 Gaussians, full circle
  * 10 UCI tasks: Boston Housing, Concrete Strength, Energy Efficiency, Kin8nm, Naval Propulsion, Power Plant, Protein Structure, Wine Quality Red, Yacht Hydrodynamics, Year Prediction MSD
* Classification
  * CIFAR-10, CIFAR-100, ImageNet, FashionMNIST, MNIST (noisy)

Note that for the UCI regression tasks, all data and the corresponding split schemes are adapted through the official repo of MC Dropout [here](https://github.com/yaringal/DropoutUncertaintyExps), except the [YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd) dataset due to its size.
  
We provide the following example to run the model on the Boston Housing regression task:

```sh
bash training_scripts/run_uci_boston.sh
```

The configuration for each of the above listed tasks (including data file location, training log and evaluation result directory settings, neural network architecture, optimization hyperparameters, etc.) are provided in the corresponding files in the `configs` directory. For each experimental run, you can find within the following 4 directories:

* `logs`: `stdout.txt` with training logs, `testmetrics.txt` with evaluation metrics
* `tensorboard`: files to track training progress
* `training_image_samples`: plots during training
* `testing_image_samples`: plots when evaluating the model

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
