#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:42:26 2020

@author: ayman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import ptan
import numpy as np
from types import SimpleNamespace
from . import utils


class A2CNet(nn.Module):
    """Double heads actor + critic network."""

    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hid_size)
        self.policy = nn.Linear(hid_size, act_size)
        self.value = nn.Linear(hid_size, 1)

    def forward(self, x):
        """Feed forward."""
        y = F.relu(self.fc1(x))
        return (self.policy(y), self.value(y))


class A3CConvNet(nn.Module):
    def __init__(self,input_shape,act_size):
        super().__init__()
        self.input_shape = input_shape
        self.act_size = act_size
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU(),
                                  )

        self._calc_conv_size()
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(self._conv_ouput_size, 512),
                                nn.ReLU(),
                                nn.Linear(512,act_size))
    def forward(self,x):
        y = x / 256
        y = ptan.agent.float32_preprocessor(y)
        y = self.conv(y)
        return self.fc(y)

    def _calc_conv_size(self):
        t = torch.zeros(1,*self.input_shape)
        out = self.conv(t).shape
        self._conv_ouput_size = np.prod(out)



class A2C_Conv(nn.Module):
    """Convlutional Neural Network to train Gym OpenAI games."""

    def __init__(self,input_shape, act_size):
        super().__init__()
        self.input_shape = input_shape
        self.act_size = act_size
        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride =2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self._calc_conv_out()
        self.p1 = nn.Linear(self._conv_out_size, 512)
        self.policy = nn.Linear(512, act_size)
        self.v1 = nn.Linear(self._conv_out_size, 512)
        self.value = nn.Linear(512, 1)

    def forward(self,x):
        x = x.float()
        x /= 256
        y = ptan.agent.float32_preprocessor(x)
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = torch.flatten(y, start_dim=1, end_dim=-1)
        p = F.relu(self.p1(y))
        v = F.relu(self.v1(y))
        return self.policy(p),self.value(v)

    def _calc_conv_out(self):
        t = torch.zeros(1,*self.input_shape)
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        out = F.relu(self.conv3(t)).shape
        self._conv_out_size = np.prod(out)


class ActorNet(nn.Module):
    """Two layers actor network."""

    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hid_size)
        self.output = nn.Linear(hid_size, act_size)

    def forward(self, x):
        """Feed forward."""
        y = F.relu(self.fc1(x))
        return self.output(y)


class CriticNet(nn.Module):
    """Two layers critic network."""

    def __init__(self, obs_size, hid_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hid_size)
        self.output = nn.Linear(hid_size, 1)

    def forward(self, x):
        """Feed forward."""
        y = F.relu(self.fc1(x))
        return self.output(y)


class MPBatchGenerator:
    """
    Iterate over environment and return batchs of (states,actions,rewards,last_states).

    Parameters
    ----------
    exp_source : ptan FirstLastExperienceSource
        .
    params : SimpleNamespace
        Contains the parameters of the environment, usually in data.py file.
    batch_size : int, optional
        The default is 256.
    baseline : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    States: float64
        .
    Actions: int64
        .
    Rewards: float64
        .
    Dones: bool
        .
    States: float64
        .
    """

    def __init__(self, exp_source, params: SimpleNamespace, batch_size: int = 256):
        self.exp_source = exp_source
        self.params = params
        self.batch_size = batch_size
        self._total_rewards = []
        self._end_episode_frames = []
        self.__reset__()

    def __reset__(self):
        """Set frame and episode counters to 0."""
        self.frame = 0
        self.episode = 0

    def __iter__(self):
        """
        Iterate over the environments and return a batch of States, Actions and Q values.

        Yields
        ------
        Numpy arraies
            States, Actions, Q_ref_values.

        """
        states, actions, rewards, dones, last_states = [], [], [], [], []
        for exp in self.exp_source:
            self.frame += 1
            new_reward = self.exp_source.pop_total_rewards()
            if new_reward:
                self._total_rewards.append(new_reward[0])
                self._end_episode_frames.append(self.frame)
                self.episode += 1
            if len(states) >= self.batch_size:
                yield (np.array(states, copy=False),
                       np.array(actions),
                       np.array(rewards),
                       np.array(dones),
                       np.array(last_states, copy=False))
                states.clear()
                actions.clear()
                rewards.clear()
                last_states.clear()
                dones.clear()
            states.append(np.array(exp.state, copy=False))
            actions.append(int(exp.action))
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            last_states.append(np.array(exp.last_state, copy=False) if
                               exp.last_state is not None else np.array(exp.state, copy=False))


class MPEpisodeGenerator:
    """
    Iterate over environment and return batchs of (states,actions,rewards,last_states).

    Parameters
    ----------
    exp_source : ptan FirstLastExperienceSource
        .
    params : SimpleNamespace
        Contains the parameters of the environment, usually in data.py file.
    batch_size : int, optional
        The default is 256.
    baseline : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    States: float64
        .
    Actions: int64
        .
    Rewards: float64
        .
    Dones: bool
        .
    States: float64
        .
    """

    def __init__(self, exp_queue):
        self.exp_queue = exp_queue
        self._total_rewards = []
        self.__reset__()

    def __reset__(self):
        """Set frame and episode counters to 0."""
        self.frame = 0
        self.episode = 0

    def __iter__(self):
        """
        Iterate over N episodes and return States, Actions and Q values.

        Yields
        ------
        Numpy arraies
            States, Actions, Q_ref_values.

        """
        while True:
            while not self.exp_queue.empty():
                data = self.exp_queue.get()
                if isinstance(data, utils.EpisodeEnd):
                    self._total_rewards.append(data.reward)
                    self.episode += 1
                    continue
                self.frame += len(data[1])
                yield data


class CategoricalSelector(ptan.actions.ActionSelector):
    """Sample Actions from Categorical distribution."""

    def __call__(self, prob):
        """
        Select actions from categorical distribution.

        Parameters
        ----------
        prob : Torch.Tensors
            Probabilities of Actions: apply softmax to network output.

        Returns
        -------
        Numpy array
            Actions sampled from Categorical distribution.

        """
        if isinstance(prob, np.ndarray):
            prob = torch.Tensor(prob)
        assert isinstance(prob, torch.Tensor)
        distribution = torch.distributions.Categorical(
            prob, validate_args=True)
        actions = distribution.sample()
        return actions.numpy()
