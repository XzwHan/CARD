import random
import logging
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch import nn
from data_loader import *


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def multiple_experiments(
        model_class,
        data_class,
        OLS_class,
        data_init_dict: dict,
        train_test_dict: dict,
        ols_init_dict: dict,
        ols_fit_dict: dict,
        model_init_dict: dict,
        model_train_dict: dict,
        model_eval_dict: dict,
        seeds=None,
        num_seeds=10,
        model_to_plot=None,
        ols_plot_dict=None,
        device="cpu"
):
    if seeds is None:
        seeds = np.arange(num_seeds)
    if model_to_plot is None:
        model_to_plot = seeds[-1]
    ols_coverages = []
    coverages = []

    for idx, seed in enumerate(seeds):
        data_init_dict.update(seed=seed)
        data = data_class(**data_init_dict)
        data.create_train_test_dataset(**train_test_dict)

        ols_init_dict.update(data=data)
        ols_model = OLS_class(**ols_init_dict)
        ols_model.fit_model(**ols_fit_dict)
        ols_model.get_test_prediction_coverage()
        ols_coverages.append(ols_model.test_prediction_coverage)
        if ols_plot_dict is not None and idx == model_to_plot:
            ols_model.plot(**ols_plot_dict)

        model_init_dict.update(seed=seed, dim_x=data.dim_x, dim_y=data.dim_y)
        model = model_class(**model_init_dict).to(device)
        print(f"\n*Create the {idx}-th model from class {type(model).__name__}, with random seed {model.seed}")

        model_train_dict.update(dataset=data)
        model.train_loop(**model_train_dict)

        model_eval_dict.update(dataset=data, make_plot=False)
        coverages.append(model.evaluate(**model_eval_dict))

        if idx == model_to_plot:
            model_eval_dict.update({"make_plot": True})
            for plot_true in (True, False):
                for plot_gen in (True, False):
                    model_eval_dict.update(plot_true=plot_true, plot_gen=plot_gen)
                    model.evaluate(**model_eval_dict)

        print(f"\n* For class {type(model).__name__} and {len(seeds)} seeds: mean coverage rate: {np.mean(coverages):.4f}, std: {np.std(coverages):.4f}")
        print(f"\n* For class {type(ols_model).__name__} and {len(seeds)} seeds: mean coverage rate: {np.mean(ols_coverages):.4f}, std: {np.std(ols_coverages):.4f}")
        return coverages, ols_coverages


# TODO: Add evaluation metrics


class SubspaceInferenceDatasetNet(nn.Sequential):
    def __init__(self, dimensions=(200, 50, 50, 50), input_dim=2, output_dim=1):
        super(SubspaceInferenceDatasetNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
            if i < len(self.dimensions) - 2:
                self.add_module('relu%d' % i, torch.nn.ReLU())

    def forward(self, x, output_features=False):
        if not output_features:
            return super().forward(x)
        else:
            print(self._modules.values())
            print(list(self._modules.values())[:-2])
            for module in list(self._modules.values())[:-3]:
                x = module(x)
                print(x.size())
            return x
        

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


def get_dataset(args, config, test_set=False, validation=False):
    data_object = None
    if config.data.dataset == 'swiss_roll':
        data = SwissRoll().sample(config.data.dataset_size)
    elif config.data.dataset == 'moons':
        data = Moons().sample(config.data.dataset_size)
    elif config.data.dataset == '8gaussians':
        data = Gaussians().sample(config.data.dataset_size - config.data.dataset_size % 8, mode=8)
    elif config.data.dataset == '25gaussians':
        data = Gaussians().sample(config.data.dataset_size - config.data.dataset_size % 25, mode=25)
    elif config.data.dataset == "uci":
        data_object = UCI_Dataset(config, args.split, validation)
        data_type = "test" if test_set else "train"
        logging.info(data_object.summary_dataset(split=data_type))
        data = data_object.return_dataset(split=data_type)
    elif config.data.dataset == "linear_regression":
        data_object = LinearDatasetWithOneX(a=config.data.a, b=config.data.b,
                                            n_samples=config.data.dataset_size,
                                            seed=args.seed,
                                            x_dict=vars(config.data.x_dict),
                                            noise_dict=vars(config.data.noise_dict),
                                            normalize_x=config.data.normalize_x,
                                            normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    elif config.data.dataset == "quadratic_regression":
        data_object = QuadraticDatasetWithOneX(a=config.data.a, b=config.data.b, c=config.data.c,
                                               n_samples=config.data.dataset_size,
                                               seed=args.seed,
                                               x_dict=vars(config.data.x_dict),
                                               noise_dict=vars(config.data.noise_dict),
                                               normalize_x=config.data.normalize_x,
                                               normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    elif config.data.dataset == "sinusoidal_regression_mdn":
        data_object = SinusoidDatasetWithOneX(n_samples=config.data.dataset_size,
                                              seed=args.seed,
                                              x_dict=vars(config.data.x_dict),
                                              noise_dict=vars(config.data.noise_dict),
                                              normalize_x=config.data.normalize_x,
                                              normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    elif config.data.dataset == "inverse_sinusoidal_regression_mdn":
        data_object = SinusoidDatasetWithOneX(n_samples=config.data.dataset_size,
                                              seed=args.seed,
                                              x_dict=vars(config.data.x_dict),
                                              noise_dict=vars(config.data.noise_dict),
                                              normalize_x=config.data.normalize_x,
                                              normalize_y=config.data.normalize_y)
        data_object.invert_xy()
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    elif config.data.dataset == "full_circle":
        data_object = CircleDatasetWithOneX(r=config.data.r, n_samples=config.data.dataset_size,
                                            seed=args.seed,
                                            x_dict=vars(config.data.x_dict),
                                            noise_dict=vars(config.data.noise_dict),
                                            normalize_x=config.data.normalize_x,
                                            normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    elif config.data.dataset == "loglog_linear_regression":
        data_object = LogLogDatasetWithOneX(a=config.data.a, b=config.data.b,
                                            n_samples=config.data.dataset_size,
                                            seed=args.seed,
                                            x_dict=vars(config.data.x_dict),
                                            noise_dict=vars(config.data.noise_dict),
                                            normalize_x=config.data.normalize_x,
                                            normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    elif config.data.dataset == "loglog_cubic_regression":
        data_object = LogLogDatasetWithOneX(a=config.data.a, b=config.data.b,
                                            n_samples=config.data.dataset_size,
                                            seed=args.seed,
                                            x_dict=vars(config.data.x_dict),
                                            noise_dict=vars(config.data.noise_dict),
                                            normalize_x=config.data.normalize_x,
                                            normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    elif config.data.dataset == "8gauss":
        data_object = CTToyDataset(n_samples=config.data.dataset_size,
                                   seed=args.seed,
                                   x_dict=vars(config.data.x_dict),
                                   noise_dict=vars(config.data.noise_dict),
                                   normalize_x=config.data.normalize_x,
                                   normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        data = data_object.test_dataset if test_set else data_object.train_dataset
    else:
        raise NotImplementedError(
            "Toy dataset options: swiss_roll, moons, 8gaussians and 25gaussians; regression data: UCI.")
    return data_object, data

