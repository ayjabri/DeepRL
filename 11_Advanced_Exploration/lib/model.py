import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
