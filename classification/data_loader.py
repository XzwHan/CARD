import torch
import os
import abc
import utils
import numpy as np
import scipy.stats as stats
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler


class Gaussians:
    """
    Gaussian mixture distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """

    def sample(self, n, noise=0.02, mode=2):
        if noise is None:
            noise = 0.02

        if mode == 2:
            scale = 2.
            centers = [
                (1, 0), (0, 1)]
            centers = [(x, y) for x, y in centers]
            temp = []
            labels = []
            for i in range(n):
                point = np.random.randn(2) * .05
                label = np.random.choice(np.arange(len(centers)))
                center = centers[label]
                point[0] += (center[0] + scale) + 5.
                point[1] += (center[1] + scale) + 10.
                temp.append(point)
                labels.append(label)
            temp = np.array(temp)
            labels = np.array(labels)
            temp /= 1.414  # stdev
        else:
            raise NotImplementedError('Toy data is a 2 mode Gaussian mixture for analysis')

        return torch.from_numpy(temp).float(), torch.from_numpy(labels)


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


class DatasetOneDimensionalX(Dataset):
    def __init__(self, n_samples, seed, label_min_max, normalize_x=False, normalize_y=False):
        super(DatasetOneDimensionalX, self).__init__(seed=seed, n_samples=n_samples)
        self.label_min = label_min_max[0]
        self.label_max = label_min_max[1]
        self.x_samples, self.y, self.y_logits, self.labels = None, None, None, None
        self.dim_x = None
        self.dim_y = None
        self.x_train, self.y_train, self.y_logits_train, self.labels_train = None, None, None, None
        self.x_test, self.y_test, self.y_logits_test, self.labels_test = None, None, None, None
        self.train_n_samples, self.test_n_samples = None, None
        self.train_dataset, self.test_dataset = None, None
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.scaler_x, self.scaler_y = None, None

    def convert_one_hot_y_to_logit(self):
        """
        Form y prototype in real number by converting one-hot label to logit.
        """
        self.y_logits = torch.logit(torch.nn.functional.normalize(
            torch.clip(self.y, min=self.label_min, max=self.label_max), p=1.0, dim=1))

    def create_train_test_dataset(self, train_ratio=0.8):
        # first create the logit version of y prototypes
        self.convert_one_hot_y_to_logit()
        # split data into train and test set
        utils.set_random_seed(self.seed)
        data_idx = np.arange(self.n_samples)
        np.random.shuffle(data_idx)
        train_size = int(self.n_samples * train_ratio)
        self.x_train, self.y_train, self.y_logits_train, self.labels_train = \
            self.x_samples[data_idx[:train_size]], self.y[data_idx[:train_size]], \
            self.y_logits[data_idx[:train_size]], self.labels[data_idx[:train_size]]
        self.x_test, self.y_test, self.y_logits_test, self.labels_test = \
            self.x_samples[data_idx[train_size:]], self.y[data_idx[train_size:]], \
            self.y_logits[data_idx[train_size:]], self.labels[data_idx[train_size:]]
        self.train_n_samples = self.x_train.shape[0]
        self.test_n_samples = self.x_test.shape[0]
        # standardize x and y if needed
        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()
        self.train_dataset = TensorDataset(self.x_train, self.y_train, self.y_logits_train, self.labels_train)
        # sort x for easier plotting purpose during test time
        if self.dim_x == 1:
            sorted_idx = torch.argsort(self.x_test, dim=0).squeeze()
            self.x_test = self.x_test[sorted_idx]
            self.y_test = self.y_test[sorted_idx]
            self.y_logits_test = self.y_logits_test[sorted_idx]
            self.labels_test = self.labels_test[sorted_idx]
        self.test_dataset = TensorDataset(self.x_test, self.y_test, self.y_logits_test, self.labels_test)

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


class GaussianMixture(DatasetOneDimensionalX):
    def __init__(self, n_samples, seed, label_min_max, dist_dict, normalize_x=False, normalize_y=False):
        super(GaussianMixture, self).__init__(
            n_samples=n_samples, seed=seed, label_min_max=label_min_max,
            normalize_x=normalize_x, normalize_y=normalize_y)
        self.means, self.sds, self.probs = dist_dict['means'], dist_dict['sds'], dist_dict['probs']
        self.sample_x_and_y(dist_dict)

    def sample_x_and_y(self, dist_dict):
        """
        :param dist_dict: contains Gaussian mixture mean, standard deviation and class probability lists.
        """
        print("Create x and y using the following parameters:")
        print(dist_dict)
        components = [torch.distributions.normal.Normal(
                        loc=self.means[i],
                        scale=self.sds[i]) for i in range(len(self.probs))]
        m = torch.distributions.categorical.Categorical(torch.tensor(self.probs))

        labels = m.sample((self.n_samples, 1))
        self.x_samples = torch.tensor([components[label.item()].sample() for label in labels]).reshape((-1, 1))
        self.y = torch.nn.functional.one_hot(labels).squeeze().float()
        self.dim_x = self.x_samples.shape[1]  # dimension of data input
        self.dim_y = self.y.shape[1]  # dimension of classification output (one-hot label)
        self.labels = labels

    def plot_samples(self):
        fig, axs = plt.subplots(1, 2, figsize=(2 * 8.5, 5), clear=True)
        axs[0].hist(self.x_samples.numpy(), bins=50, density=True)
        x = np.linspace(-3, 4, 1000)
        pdf_all_components = np.array(
            [self.probs[i] * stats.norm.pdf(x, self.means[i], self.sds[i]) for i in range(len(self.probs))]).sum(0)
        axs[0].plot(x, pdf_all_components)
        axs[0].set_title('Sample Histogram and PDF', fontsize=14)
        axs[1].hist([self.x_samples[self.labels == 0].numpy(),
                     self.x_samples[self.labels == 1].numpy(),
                     self.x_samples[self.labels == 2].numpy()],
                    bins=50, density=False, color=['r', 'g', 'b'], label=[0, 1, 2])
        axs[1].legend(loc='best')
        axs[1].set_title('Histogram with Labels', fontsize=14)
        fig.suptitle('Mixture Gaussian Classification', fontsize=16);

    def compute_class_posterior(self, x):
        weighted_pdf_components = [
            self.probs[i] * stats.norm.pdf(x, self.means[i], self.sds[i]) for i in range(len(self.probs))]
        denominator = np.array(weighted_pdf_components).sum(0)
        x_class_posterior = [(class_i_weighted_pdf/denominator).flatten()
                             for class_i_weighted_pdf in weighted_pdf_components]
        return x_class_posterior


class AddGaussianNoise(object):
    """
    Add standard Gaussian noise to MNIST dataset to create noisy MNIST dataset.
    From https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.makedirs('./data')

    import matplotlib.pyplot as plt

    plt.style.use('ggplot')

    x, y = Gaussians().sample(10000, mode=2)
    x = x.data.numpy()
    y_vec = torch.nn.functional.one_hot(y).data.numpy()
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].scatter(x[:, 0], x[:, 1], s=5, c=y);
    axs[0].set_title(r'data distribution')
    axs[1].scatter(y_vec[:, 0], y_vec[:, 1], s=5, c=y);
    axs[1].set_title(r'label distribution')
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'gaussians.pdf'))
