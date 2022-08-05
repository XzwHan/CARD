import torch
import os
import utils
import numpy as np
import pandas as pd
import abc
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.preprocessing import StandardScaler


class SwissRoll:
    """
    Swiss roll distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """

    def sample(self, n, noise=0.5):
        if noise is None:
            noise = 0.5
        return torch.from_numpy(
            make_swiss_roll(n_samples=n, noise=noise)[0][:, [0, 2]].astype('float32') / 5.)


class Moons:
    """
    Double moons distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """

    def sample(self, n, noise=0.02):
        if noise is None:
            noise = 0.02
        temp = make_moons(n_samples=n, noise=noise)[0].astype('float32')
        return torch.from_numpy(temp / abs(temp).max())


class Gaussians:
    """
    Gaussian mixture distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """

    def sample(self, n, noise=0.02, mode=8):
        if noise is None:
            noise = 0.02

        if mode == 8:
            scale = 2.
            centers = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            centers = [(scale * x, scale * y) for x, y in centers]
            temp = []
            labels = []
            for i in range(n):
                point = np.random.randn(2) * .02
                label = np.random.choice(np.arange(len(centers)))
                center = centers[label]
                point[0] += center[0]
                point[1] += center[1]
                temp.append(point)
                labels.append(label)
            temp = np.array(temp, dtype='float32')
            labels = np.array(labels)
            temp /= 1.414  # stdev
        elif mode == 25:
            temp = []
            labels = []
            for i in range(int(n / 25)):
                label = 0
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2) * 0.05
                        point[0] += 2 * x
                        point[1] += 2 * y
                        temp.append(point)
                        labels.append(label)
                        label += 1
            temp = np.array(temp, dtype='float32')
            labels = np.array(labels)
            rand_idx = np.arange(n)
            np.random.shuffle(rand_idx)
            temp = temp[rand_idx] / 2.828  # stdev
            labels = labels[rand_idx]
        return torch.from_numpy(temp)


def _get_index_train_test_path(data_directory_path, split_num, train=True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return os.path.join(data_directory_path, "index_train_" + str(split_num) + ".txt")
    else:
        return os.path.join(data_directory_path, "index_test_" + str(split_num) + ".txt")


def onehot_encode_cat_feature(X, cat_var_idx_list):
    """
    Apply one-hot encoding to the categorical variable(s) in the feature set,
        specified by the index list.
    """
    # select numerical features
    X_num = np.delete(arr=X, obj=cat_var_idx_list, axis=1)
    # select categorical features
    X_cat = X[:, cat_var_idx_list]
    X_onehot_cat = []
    for col in range(X_cat.shape[1]):
        X_onehot_cat.append(pd.get_dummies(X_cat[:, col], drop_first=True))
    X_onehot_cat = np.concatenate(X_onehot_cat, axis=1).astype(np.float32)
    dim_cat = X_onehot_cat.shape[1]  # number of categorical feature(s)
    X = np.concatenate([X_num, X_onehot_cat], axis=1)
    return X, dim_cat


def preprocess_uci_feature_set(X, config):
    """
    Obtain preprocessed UCI feature set X (one-hot encoding applied for categorical variable)
        and dimension of one-hot encoded categorical variables.
    """
    dim_cat = 0
    task_name = config.data.dir
    if config.data.one_hot_encoding:
        if task_name == 'bostonHousing':
            X, dim_cat = onehot_encode_cat_feature(X, [3])
        elif task_name == 'energy':
            X, dim_cat = onehot_encode_cat_feature(X, [4, 6, 7])
        elif task_name == 'naval-propulsion-plant':
            X, dim_cat = onehot_encode_cat_feature(X, [0, 1, 8, 11])
        else:
            pass
    return X, dim_cat


############################
### UCI regression tasks ###
class UCI_Dataset(object):
    def __init__(self, config, split, validation=False):
        # global variables for reading data files
        _DATA_DIRECTORY_PATH = os.path.join(config.data.data_root, config.data.dir, "data")
        _DATA_FILE = os.path.join(_DATA_DIRECTORY_PATH, "data.txt")
        _INDEX_FEATURES_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_features.txt")
        _INDEX_TARGET_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_target.txt")
        _N_SPLITS_FILE = os.path.join(_DATA_DIRECTORY_PATH, "n_splits.txt")

        # set random seed 1 -- same setup as MC Dropout
        utils.set_random_seed(1)

        # load the data
        data = np.loadtxt(_DATA_FILE)
        # load feature and target indices
        index_features = np.loadtxt(_INDEX_FEATURES_FILE)
        index_target = np.loadtxt(_INDEX_TARGET_FILE)
        # load feature and target as X and y
        X = data[:, [int(i) for i in index_features.tolist()]].astype(np.float32)
        y = data[:, int(index_target.tolist())].astype(np.float32)
        # preprocess feature set X
        X, dim_cat = preprocess_uci_feature_set(X=X, config=config)
        self.dim_cat = dim_cat

        # load the indices of the train and test sets
        index_train = np.loadtxt(_get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=False))

        # read in data files with indices
        x_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]].reshape(-1, 1)
        x_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]].reshape(-1, 1)

        # split train set further into train and validation set for hyperparameter tuning
        if validation:
            num_training_examples = int(config.diffusion.nonlinear_guidance.train_ratio * x_train.shape[0])
            x_test = x_train[num_training_examples:, :]
            y_test = y_train[num_training_examples:]
            x_train = x_train[0:num_training_examples, :]
            y_train = y_train[0:num_training_examples]

        self.x_train = x_train if type(x_train) is torch.Tensor else torch.from_numpy(x_train)
        self.y_train = y_train if type(y_train) is torch.Tensor else torch.from_numpy(y_train)
        self.x_test = x_test if type(x_test) is torch.Tensor else torch.from_numpy(x_test)
        self.y_test = y_test if type(y_test) is torch.Tensor else torch.from_numpy(y_test)

        self.train_n_samples = x_train.shape[0]
        self.train_dim_x = self.x_train.shape[1]  # dimension of training data input
        self.train_dim_y = self.y_train.shape[1]  # dimension of training regression output

        self.test_n_samples = x_test.shape[0]
        self.test_dim_x = self.x_test.shape[1]  # dimension of testing data input
        self.test_dim_y = self.y_test.shape[1]  # dimension of testing regression output

        self.normalize_x = config.data.normalize_x
        self.normalize_y = config.data.normalize_y
        self.scaler_x, self.scaler_y = None, None

        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()

    def normalize_train_test_x(self):
        """
        When self.dim_cat > 0, we have one-hot encoded number of categorical variables,
            on which we don't conduct standardization. They are arranged as the last
            columns of the feature set.
        """
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        if self.dim_cat == 0:
            self.x_train = torch.from_numpy(
                self.scaler_x.fit_transform(self.x_train).astype(np.float32))
            self.x_test = torch.from_numpy(
                self.scaler_x.transform(self.x_test).astype(np.float32))
        else:  # self.dim_cat > 0
            x_train_num, x_train_cat = self.x_train[:, :-self.dim_cat], self.x_train[:, -self.dim_cat:]
            x_test_num, x_test_cat = self.x_test[:, :-self.dim_cat], self.x_test[:, -self.dim_cat:]
            x_train_num = torch.from_numpy(
                self.scaler_x.fit_transform(x_train_num).astype(np.float32))
            x_test_num = torch.from_numpy(
                self.scaler_x.transform(x_test_num).astype(np.float32))
            self.x_train = torch.from_numpy(np.concatenate([x_train_num, x_train_cat], axis=1))
            self.x_test = torch.from_numpy(np.concatenate([x_test_num, x_test_cat], axis=1))

    def normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = torch.from_numpy(
            self.scaler_y.fit_transform(self.y_train).astype(np.float32))
        self.y_test = torch.from_numpy(
            self.scaler_y.transform(self.y_test).astype(np.float32))

    def return_dataset(self, split="train"):
        if split == "train":
            train_dataset = torch.cat((self.x_train, self.y_train), dim=1)
            return train_dataset
        else:
            test_dataset = torch.cat((self.x_test, self.y_test), dim=1)
            return test_dataset

    def summary_dataset(self, split="train"):
        if split == "train":
            return {'n_samples': self.train_n_samples, 'dim_x': self.train_dim_x, 'dim_y': self.train_dim_y}
        else:
            return {'n_samples': self.test_n_samples, 'dim_x': self.test_dim_x, 'dim_y': self.test_dim_y}


def compute_y_noiseless_mean(dataset, x_test_batch, true_function='linear'):
    """
    Compute the mean of y with the ground truth data generation function.
    """
    if true_function == 'linear':
        y_true_mean = dataset.a + dataset.b * x_test_batch
    elif true_function == 'quadratic':
        y_true_mean = dataset.a * x_test_batch.pow(2) + dataset.b * x_test_batch + dataset.c
    elif true_function == 'loglinear':
        y_true_mean = (dataset.a + dataset.b * x_test_batch).exp()
    elif true_function == 'loglog':
        y_true_mean = (np.log(dataset.a) + dataset.b * x_test_batch.log()).exp()
    elif true_function == 'mdnsinusoidal':
        y_true_mean = x_test_batch + 0.3 * torch.sin(2 * np.pi * x_test_batch)
    elif true_function == 'sinusoidal':
        y_true_mean = x_test_batch * torch.sin(x_test_batch)
    else:
        raise NotImplementedError('We don\'t have such data generation scheme for toy example.')
    return y_true_mean.numpy()


class Dataset(object):
    def __init__(self, seed, n_samples):
        self.seed = seed
        self.n_samples = n_samples
        self.eps_samples = None
        utils.set_random_seed(self.seed)

    @abc.abstractmethod
    def create_train_test_dataset(self):
        pass

    def create_noises(self, noise_dict):
        """
        Ref: https://pytorch.org/docs/stable/distributions.html
        :param noise_dict: {"noise_type": "norm", "loc": 0., "scale": 1.}
        """
        print("Create noises using the following parameters:")
        print(noise_dict)
        noise_type = noise_dict.get("noise_type", "norm")
        if noise_type == "t":
            dist = torch.distributions.studentT.StudentT(df=noise_dict.get("df", 10.), loc=noise_dict.get("loc", 0.0),
                                                         scale=noise_dict.get("scale", 1.0))
        elif noise_type == "unif":
            dist = torch.distributions.uniform.Uniform(low=noise_dict.get("low", 0.), high=noise_dict.get("high", 1.))
        elif noise_type == "Chi2":
            dist = torch.distributions.chi2.Chi2(df=noise_dict.get("df", 10.))
        elif noise_type == "Laplace":
            dist = torch.distributions.laplace.Laplace(loc=noise_dict.get("loc", 0.), scale=noise_dict.get("scale", 1.))
        else:  # noise_type == "norm"
            dist = torch.distributions.normal.Normal(loc=noise_dict.get("loc", 0.), scale=noise_dict.get("scale", 1.))

        self.eps_samples = dist.sample((self.n_samples, 1))


class DatasetWithOneX(Dataset):
    def __init__(self, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        super(DatasetWithOneX, self).__init__(seed=seed, n_samples=n_samples)
        self.x_dict = x_dict
        self.x_samples = self.sample_x(self.x_dict)
        self.dim_x = self.x_samples.shape[1]  # dimension of data input
        self.y = self.create_y_from_one_x(noise_dict=noise_dict)
        self.dim_y = self.y.shape[1]  # dimension of regression output
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.train_n_samples, self.test_n_samples = None, None
        self.train_dataset, self.test_dataset = None, None
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.scaler_x, self.scaler_y = None, None

    def sample_x(self, x_dict):
        """
        :param x_dict: {"dist_type": "unif", "low": 0., "high": 1.}
        """
        print("Create x using the following parameters:")
        print(x_dict)
        dist_type = x_dict.get("dist_type", "unif")
        if dist_type == "norm":
            dist = torch.distributions.normal.Normal(loc=x_dict.get("loc", 0.), scale=x_dict.get("scale", 1.))
        else:
            dist = torch.distributions.uniform.Uniform(low=x_dict.get("low", 0.), high=x_dict.get("high", 1.))

        return dist.sample((self.n_samples, 1))

    def create_y_from_one_x(self, noise_dict):
        if self.eps_samples is None:
            n_samples_temp = self.n_samples
            if type(noise_dict.get("scale", 1.)) == torch.Tensor:
                self.n_samples = 1
            self.create_noises(noise_dict)
            if self.n_samples == 1:
                self.eps_samples = self.eps_samples[0]
                self.n_samples = n_samples_temp

    def create_train_test_dataset(self, train_ratio=0.8):
        utils.set_random_seed(self.seed)
        data_idx = np.arange(self.n_samples)
        np.random.shuffle(data_idx)
        train_size = int(self.n_samples * train_ratio)
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.x_samples[data_idx[:train_size]], self.y[data_idx[:train_size]], \
            self.x_samples[data_idx[train_size:]], self.y[data_idx[train_size:]]
        self.train_n_samples = self.x_train.shape[0]
        self.test_n_samples = self.x_test.shape[0]
        # standardize x and y if needed
        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()
        self.train_dataset = torch.cat((self.x_train, self.y_train), dim=1)
        # sort x for easier plotting purpose during test time
        if self.dim_x == 1:
            sorted_idx = torch.argsort(self.x_test, dim=0).squeeze()
            self.x_test = self.x_test[sorted_idx]
            self.y_test = self.y_test[sorted_idx]
        self.test_dataset = torch.cat((self.x_test, self.y_test), dim=1)

    def normalize_train_test_x(self):
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        self.x_train = torch.from_numpy(
            self.scaler_x.fit_transform(self.x_train).astype(np.float32))
        self.x_test = torch.from_numpy(
            self.scaler_x.transform(self.x_test).astype(np.float32))

    def normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = torch.from_numpy(
            self.scaler_y.fit_transform(self.y_train).astype(np.float32))
        self.y_test = torch.from_numpy(
            self.scaler_y.transform(self.y_test).astype(np.float32))


class LinearDatasetWithOneX(DatasetWithOneX):
    def __init__(self, a, b, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.a = a
        self.b = b
        super(LinearDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        super().create_y_from_one_x(noise_dict)
        y = self.a + self.b * self.x_samples + self.eps_samples
        return y


class QuadraticDatasetWithOneX(DatasetWithOneX):
    def __init__(self, a, b, c, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.a = a
        self.b = b
        self.c = c
        super(QuadraticDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        super().create_y_from_one_x(noise_dict)
        y = self.a * self.x_samples.pow(2) + self.b * self.x_samples + self.c + self.eps_samples
        return y


class CircleDatasetWithOneX(DatasetWithOneX):
    def __init__(self, r, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.r = r
        self.theta = torch.rand((n_samples, 1)) * 2 * np.pi
        x_dict.update(r=r)
        super(CircleDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y
        )

    def create_y_from_one_x(self, noise_dict):
        super().create_y_from_one_x(noise_dict)
        r_samples = self.r + self.eps_samples
        self.x_samples, y = r_samples * torch.cos(self.theta), r_samples * torch.sin(self.theta)
        self.dim_x = self.x_samples.shape[1]
        return y


class LogLinearDatasetWithOneX(DatasetWithOneX):
    def __init__(self, a, b, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.a = a
        self.b = b
        super(LogLinearDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        # log(y) = a + b * x + eps
        super().create_y_from_one_x(noise_dict)
        logy = self.a + self.b * self.x_samples + self.eps_samples
        return logy.exp()


class LogLogDatasetWithOneX(DatasetWithOneX):
    def __init__(self, a, b, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.a = a
        self.b = b
        super(LogLogDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        # log(y) = log(a) + b * log(x) + eps
        super().create_y_from_one_x(noise_dict)
        logy = np.log(self.a) + self.b * self.x_samples.log() + self.eps_samples
        return logy.exp()


class LinearDatasetWithStdIncreaseWithX(LinearDatasetWithOneX):
    def __init__(self, a, b, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        super(LinearDatasetWithStdIncreaseWithX, self).__init__(
            a=a, b=b, n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        # std of eps increases with abs(x), fix eps to be normal noises
        noise_dict.update(noise_type="norm", scale=noise_dict.get("scale", 1.) * self.x_samples.abs())
        return super().create_y_from_one_x(noise_dict)


class SinusoidDatasetWithOneX(DatasetWithOneX):
    """
    Both "Snelson" Dataset and "OAT-1D" Dataset are sinusoid curve.
    Currently use function from Fig. 4 of Mixture Density Networks.
    """

    def __init__(self, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        super(SinusoidDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        # y = x + 0.3 * sin(2 * pi * x) + eps
        super().create_y_from_one_x(noise_dict)
        y = self.x_samples + 0.3 * torch.sin(2 * np.pi * self.x_samples) + self.eps_samples
        return y

    def invert_xy(self):
        """
        Swap x and y to have a one-to-many mapping, like in MDN paper.
        """
        temp_x = self.y
        temp_dim_x = self.dim_y
        self.y = self.x_samples
        self.dim_y = self.dim_x
        self.x_samples = temp_x
        self.dim_x = temp_dim_x


class SubspaceInferenceDatasetWithOneX(DatasetWithOneX):
    def __init__(self, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.x_ranges = [[-7.2, -4.8], [-1.2, 1.2], [4.8, 7.2]]
        x_dict.update(x_ranges=self.x_ranges)
        super(SubspaceInferenceDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        """
        Ref: https://arxiv.org/pdf/1907.07504.pdf, Section 5.1
        The network takes two inputs, x and x ** 2
        """
        super().create_y_from_one_x(noise_dict)
        x_ranges_len = [x[1] - x[0] for x in self.x_ranges]
        x_ranges_total_len = sum(x_ranges_len)
        x_ranges_num_samples = [int(x / x_ranges_total_len * self.n_samples) for x in x_ranges_len]
        self.x_samples = torch.tensor([])
        for idx, x_range in enumerate(self.x_ranges):
            self.x_samples = torch.cat([
                self.x_samples,
                torch.distributions.uniform.Uniform(
                    low=self.x_ranges[idx][0], high=self.x_ranges[idx][1]).sample((x_ranges_num_samples[idx], 1))
            ], axis=0)
        self.dim_x = self.x_samples.shape[1]
        y = utils.SubspaceInferenceDatasetNet().forward(
            x=torch.cat([self.x_samples, self.x_samples ** 2], axis=1)) + self.eps_samples
        return y


class CTToyDataset(DatasetWithOneX):
    def __init__(self, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        super(CTToyDataset, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y
        )

    def create_y_from_one_x(self, noise_dict):
        super().create_y_from_one_x(noise_dict)
        X = self.load_data(name=self.x_dict.get("ct_toy_name", '8gaussians'))
        self.x_samples = X[:, :1]
        self.dim_x = self.x_samples.shape[1]
        y = X[:, 1:]  # + self.eps_samples
        # TODO: do we need adding noise to y
        return y

    def load_data(self, name):
        N = self.n_samples
        if name == 'swiss_roll':
            temp = make_swiss_roll(n_samples=N, noise=0.05)[0][:, (0, 2)]
            temp /= abs(temp).max()
        elif name == 'half_moons':
            temp = make_moons(n_samples=N, noise=0.02)[0]
            temp /= abs(temp).max()
        elif name == '2gaussians':
            scale = 2.
            centers = [
                (1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            centers = [(scale * x, scale * y) for x, y in centers]
            temp = []
            for i in range(N):
                point = np.random.randn(2) * .02
                center = centers[np.random.choice(np.arange(len(centers)))]
                point[0] += center[0]
                point[1] += center[1]
                temp.append(point)
            temp = np.array(temp, dtype='float32')
            temp /= 1.414  # stdev
        elif name == '8gaussians':
            scale = 2.
            centers = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            centers = [(scale * x, scale * y) for x, y in centers]
            temp = []
            for i in range(N):
                point = np.random.randn(2) * .1  # .02
                center = centers[np.random.choice(np.arange(len(centers)))]
                point[0] += center[0]
                point[1] += center[1]
                temp.append(point)
            temp = np.array(temp, dtype='float32')
            temp /= 1.414  # stdev
        elif name == '25gaussians':
            temp = []
            for i in range(int(N / 25)):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2) * 0.05
                        point[0] += 2 * x
                        point[1] += 2 * y
                        temp.append(point)
            temp = np.array(temp, dtype='float32')
            np.random.shuffle(temp)
            temp /= 2.828  # stdev
        elif name == 'circle':
            temp, y = make_circles(n_samples=N, noise=0.05)
            temp = temp[np.argwhere(y == 0).squeeze(), :]
        elif name == 's_curve':
            temp = make_s_curve(n_samples=N, noise=0.02)[0]  # n_samples=500
            temp = np.stack([temp[:, 0], temp[:, 2]], axis=1)
        else:
            raise Exception(
                ("Dataset not found: name must be 'swiss_roll', 'half_moons', " +
                 "'circle', 's_curve', '8gaussians' or '25gaussians'."))
        X = torch.from_numpy(temp).float()
        return X


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.makedirs('./data')

    import matplotlib.pyplot as plt

    sampler = SwissRoll()
    x = sampler.sample(10000).data.numpy()
    plt.close('all')
    fig = plt.figure(figsize=(5, 5))
    _ = plt.hist2d(x[:, 0], x[:, 1], 200, )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'swiss_roll.pdf'))

    sampler = Moons()
    x = sampler.sample(10000).data.numpy()
    plt.close('all')
    fig = plt.figure(figsize=(5, 5))
    _ = plt.hist2d(x[:, 0], x[:, 1], 200, )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'moons.pdf'))

    sampler = Gaussians()
    x = sampler.sample(10000, mode=8).data.numpy()
    plt.close('all')
    fig = plt.figure(figsize=(5, 5))
    _ = plt.hist2d(x[:, 0], x[:, 1], 200, )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join('data', '8gaussians.pdf'))

    sampler = Gaussians()
    x = sampler.sample(10000, mode=25).data.numpy()
    plt.close('all')
    fig = plt.figure(figsize=(5, 5))
    _ = plt.hist2d(x[:, 0], x[:, 1], 200, )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join('data', '25gaussians.pdf'))
