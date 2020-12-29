#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 05:42:48 2020

@author: ayman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ptan.agent import BaseAgent, float32_preprocessor
from collections import namedtuple


class ActorNet(nn.Module):
    def __init__(self, obs_size, act_size, high_action=1):
        super().__init__()
        self.high_action = high_action

        self.base = nn.Linear(obs_size, 400)
        self.fc1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, 300)
        self.actions = nn.Linear(300, act_size)

    def forward(self, x):
        y = F.relu(self.base(x))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = torch.tanh(self.actions(y))
        return self.high_action * y


class CriticNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.obs_size = params.obs_size
        self.act_size = params.act_size
        self.vmin = params.vmin
        self.vmax = params.vmax
        self.natoms = params.natoms
        self.dz = (self.vmax - self.vmin)/(self.natoms - 1)
        self.batch_size = params.batch_size

        self.base = nn.Linear(self.obs_size, 400)
        self.fc1 = nn.Linear(400 + self.act_size, 300)
        self.fc2 = nn.Linear(300, 200)
        self.out = nn.Linear(200, self.natoms)
        self.softmax = nn.Softmax()

        line = torch.linspace(self.vmin, self.vmax, self.natoms)
        self.register_buffer('line_support', line)

    def forward(self, s, a):
        base = self.base(s)
        y = torch.cat([base, a], dim=1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        return self.out(y)

    def dist_to_q(self, dist):
        weighted_q = torch.softmax(dist, dim=1) * self.line_support
        return weighted_q.sum(dim=1)


class DDPGAgent(BaseAgent):
    def __init__(self, actor_net, epsilon=0.3, preprocessor=None, max_action=1, min_action=-1):
        self.epsilon = epsilon
        self.model = actor_net
        self.max_action = max_action
        self.min_action = min_action
        if preprocessor is None:
            self.preprocessor = float32_preprocessor
        else:
            self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        states_v = self.preprocessor(states)
        actions_ = self.model(states_v).data.numpy()
        actions_ += np.random.normal(size=actions_.shape)
        actions_np = np.clip(actions_, self.min_action, self.max_action)
        return actions_np, agent_states
