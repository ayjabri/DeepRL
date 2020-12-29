# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)


class C51Net(nn.Module):
    '''
    Definition: C51Net(obs_size, act_size, Vmin, Vmax, Natoms)

    Categorical Linear Network with three fully connected layers
    Spits out n actions x n atoms

    Parameters
    ----------
    obs_size : int

    act_size : int

    Vmin, Vmax, Natoms: rewards range and number of atoms used to project the probability distribution
    '''

    def __init__(self, obs_size, act_size, Vmin, Vmax, Natoms):
        super().__init__()
        self.natoms = Natoms
        hid2 = int(act_size * Natoms * 2)
        hid1 = int(hid2 * 1.5)
        self.base = nn.Sequential(nn.Linear(obs_size, hid1),
                                  nn.ReLU(),
                                  nn.Linear(hid1, hid2),
                                  nn.ReLU(),
                                  nn.Linear(hid2, Natoms * act_size),
                                  )

        self.register_buffer('support', torch.linspace(Vmin, Vmax, Natoms))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_size = x.shape[0]
        return self.base(x).view(batch_size, -1, self.natoms)

    def both(self, x):
        out = self(x)
        probs = self.softmax(out)
        weights = probs * self.support
        qvals = weights.sum(dim=2)
        return out, qvals

    def qvals(self, x):
        return self.both(x)[1]


class DuelDQNet(nn.Module):
    """
    Definition: DuelDQNet(obs_size, act_size, hid_size=256)
    """

    def __init__(self, obs_size, act_size, hid_size=256):
        super().__init__()
        self.base = nn.Linear(obs_size, hid_size)
        self.val = nn.Linear(hid_size, 1)
        self.adv = nn.Linear(hid_size, act_size)

    def forward(self, x):
        x = F.relu(self.base(x))
        val = self.val(x)
        adv = self.adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class BigRamDuel(nn.Module):
    """
    Definition: DuelDQNet(obs_size, act_size)
    """

    def __init__(self, obs_size, act_size):
        super().__init__()
        self.base = nn.Linear(obs_size, 256)
        self.fc1 = nn.Linear(256, 256)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(128, 64)
        self.val = nn.Linear(64, 1)
        self.adv = nn.Linear(64, act_size)

    def forward(self, x):
        x /= 255
        out = F.relu(self.base(x))
        out = F.relu(self.fc1(out))
        out = self.drop1(out)
        out = F.relu(self.fc2(out))
        out = self.drop2(out)
        out = F.relu(self.fc3(out))
        val = self.val(out)
        adv = self.adv(out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class NoisyDQNet(nn.Module):
    """
    Definition: NoisyDQNet(obs_size,act_size,hid_size=348)
    """

    def __init__(self, obs_size, act_size, hid_size=348):
        super().__init__()
        self.base = nn.Linear(obs_size, hid_size)
        self.noisy1 = NoisyLinear(hid_size, int(hid_size/2))
        self.noisy2 = NoisyLinear(int(hid_size/2), act_size)

    def forward(self, x):
        x = F.relu(self.base(x))
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)


class DQNet(nn.Module):
    """
    Definition: DQNet(obs_size,act_size,hid_size=256)

    Regular Deep Q Network with three Linear layers
    """

    def __init__(self, obs_size, act_size, hid_size=256):
        super().__init__()
        self.fc_in = nn.Linear(obs_size, hid_size)
        self.fc_out = nn.Linear(hid_size, act_size)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        return self.fc_out(x)
