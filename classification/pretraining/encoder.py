import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50


# adapted from https://colab.research.google.com/github/kjamithash/Pytorch_DeepLearning_Experiments/blob/master
# /FashionMNIST_ResNet_TransferLearning.ipynb
class ResNetEmbedding(nn.Module):
    def __init__(self, arch, in_channels=1):
        super(ResNetEmbedding, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        if arch == 'resnet50':
            self.model = resnet50(pretrained=True)
        elif arch == 'resnet18':
            self.model = resnet18(pretrained=True)

        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.featdim = self.model.fc.in_features
        # # Change the output layer to output 10 classes instead of 1000 classes
        # self.model.fc = nn.Linear(self.featdim, 10)

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.f = []
        num_class = config.data.num_classes
        arch = 'resnet18'
        in_channels = 1 if config.data.dataset != "CIFAR10" else 3
        backbone = ResNetEmbedding(arch, in_channels)
        # obtain NN modules before the output linear layer
        for name, module in backbone.model.named_children():
            if name != 'fc':
                self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.featdim = backbone.featdim
        # output linear layer without bias, as class prototype matrix
        self.g = nn.Linear(self.featdim, num_class, bias=False)

    def forward_feature(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature

    def forward(self, x, return_feature=False):
        feature = self.forward_feature(x)
        out = self.g(feature)
        if return_feature:
            return feature, out
        else:
            return out

