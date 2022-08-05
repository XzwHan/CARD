import random
import math
import numpy as np
import argparse
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import transforms


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def sizeof_fmt(num, suffix='B'):
    """
    https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# print("Check memory usage of different variables:")
# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                          key=lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def get_optimizer(config_optim, parameters):
    if config_optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay,
                          betas=(config_optim.beta1, 0.999), amsgrad=config_optim.amsgrad,
                          eps=config_optim.eps)
    elif config_optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay)
    elif config_optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config_optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config_optim.optimizer))


def get_optimizer_and_scheduler(config, parameters, epochs, init_epoch):
    scheduler = None
    optimizer = get_optimizer(config, parameters)
    if hasattr(config, "T_0"):
        T_0 = config.T_0
    else:
        T_0 = epochs // (config.n_restarts + 1)
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=T_0,
                                                                   T_mult=config.T_mult,
                                                                   eta_min=config.eta_min,
                                                                   last_epoch=-1)
        scheduler.last_epoch = init_epoch - 1
    return optimizer, scheduler


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.training.warmup_epochs:
        lr = config.optim.lr * epoch / config.training.warmup_epochs
    else:
        lr = config.optim.min_lr + (config.optim.lr - config.optim.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - config.training.warmup_epochs) / (
                     config.training.n_epochs - config.training.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_dataset(args, config):
    data_object = None
    if config.data.dataset == 'toy':
        tr_x, tr_y = Gaussians().sample(config.data.dataset_size)
        te_x, te_y = Gaussians().sample(config.data.dataset_size)
        train_dataset = torch.utils.data.TensorDataset(tr_x, tr_y)
        test_dataset = torch.utils.data.TensorDataset(te_x, te_y)
    elif config.data.dataset == 'MNIST':
        if config.data.noisy:
            # noisy MNIST as in Contextual Dropout --  no normalization, add standard Gaussian noise
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
                AddGaussianNoise(0., 1.)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = torchvision.datasets.MNIST(root=config.data.dataroot, train=True, download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=config.data.dataroot, train=False, download=True,
                                                  transform=transform)
    elif config.data.dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST(root=config.data.dataroot, train=True, download=True,
                                                          transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=config.data.dataroot, train=False, download=True,
                                                         transform=transform)
    elif config.data.dataset == "CIFAR10":
        data_norm_mean, data_norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        # data_norm_mean, data_norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=data_norm_mean, std=data_norm_std)
             ])
        train_dataset = torchvision.datasets.CIFAR10(root=config.data.dataroot, train=True,
                                                     download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=config.data.dataroot, train=False,
                                                    download=True, transform=transform)
    elif config.data.dataset == "CIFAR100":
        data_norm_mean, data_norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        # data_norm_mean, data_norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=data_norm_mean, std=data_norm_std)
             ])
        train_dataset = torchvision.datasets.CIFAR100(root=config.data.dataroot, train=True,
                                                      download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=config.data.dataroot, train=False,
                                                     download=True, transform=transform)
    elif config.data.dataset == "gaussian_mixture":
        data_object = GaussianMixture(n_samples=config.data.dataset_size,
                                      seed=args.seed,
                                      label_min_max=config.data.label_min_max,
                                      dist_dict=vars(config.data.dist_dict),
                                      normalize_x=config.data.normalize_x,
                                      normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        train_dataset, test_dataset = data_object.train_dataset, data_object.test_dataset
    else:
        raise NotImplementedError(
            "Options: toy (classification of two Gaussian), MNIST, FashionMNIST, CIFAR10.")
    return data_object, train_dataset, test_dataset


# ------------------------------------------------------------------------------------
# Revised from timm == 0.3.2
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
# output: the prediction from diffusion model (B x n_classes)
# target: label indices (B)
# ------------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.size()[1])
    # output = torch.softmax(-(output - 1)**2,  dim=-1)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def cast_label_to_one_hot_and_prototype(y_labels_batch, config, return_prototype=True):
    """
    y_labels_batch: a vector of length batch_size.
    """
    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=config.data.num_classes).float()
    if return_prototype:
        label_min, label_max = config.data.label_min_max
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch
