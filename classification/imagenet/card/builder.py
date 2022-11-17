# Built by Huangjie Zheng
# Modified from MoCo model: https://github.com/facebookresearch/moco/blob/main/moco/builder.py 

import torch
import torch.nn as nn


class CARD(nn.Module):
    """
    Build a CARD model.
    """

    def __init__(self, model_phi, num_classes=100, n_steps=1000):
        """
        num_classes: # classes for the final prediction
        n_steps: # timesteps in diffusion process (default: 1000)
        """
        super(CARD, self).__init__()

        # create prior network
        self.model_phi = model_phi(num_classes=num_classes)

        # build the Unet
        prev_dim = self.model_phi.fc.weight.shape[1]
        self.model_theta = ConditionalModel(y_dim=num_classes, feature_dim=prev_dim, n_steps=n_steps, guidance=True)

        # split prior as encoder and linear head
        self.linear = self.model_phi.fc
        self.model_phi.fc = nn.Identity()

    def forward_features(self, x):
        return self.model_phi(x)

    def forward(self, x, y, t, yhat=None, mode='theta'):
        """
        Input:
            x: data (covariates) in mode phi; feature of data in mode theta 
            y: prediction at step t
            t: timestep index
        Output:
            predicted y_t-1 from diffusion part and predicted sample from phi
        """

        # model_phi: compute features for one view and its prediction
        if mode == 'phi':
            x = self.forward_features(x)  # NxD
            yhat = self.linear(x)  # NxC
            return x, yhat.softmax(dim=-1)
        elif mode == 'theta':
            y_next = self.model_theta(x, y, t, yhat)  # NxC
            return y_next


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, y_dim=100, feature_dim=2048, n_steps=1000, guidance=False):
        super(ConditionalModel, self).__init__()

        self.guidance = guidance

        # feature projection
        self.linx = nn.Linear(feature_dim, feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.lin3 = nn.Linear(feature_dim, y_dim)

    def forward(self, z, y, t, yhat=None):
        if self.guidance:
            y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.linx(z) * y
        y = self.lin2(y, t)
        return self.lin3(y)
