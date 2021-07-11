#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:58:02 2020

@author: Ayman Al Jabri
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import ptan
import numpy as np
from types import SimpleNamespace
# from collections import deque, namedtuple


class PGNet(nn.Module):
    r"""Plain vanilla policy gradient network."""

    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()
        # self.fc1 = nn.Linear(obs_size, hid_size)
        self.fc1 = nn.GRUCell(obs_size, hid_size)
        self.output = nn.Linear(hid_size, act_size, bias=True)

    def forward(self, x):
        """Feed forward."""
        y = F.relu(self.fc1(x))
        return self.output(y)


class RNNPG(nn.Module):
    r"""Recurrent plain vanila policy gradient model."""

    def __init__(self, obs_size, act_size, sequence=5, hid_size=128, num_layers=2):
        super().__init__()
        self.sequence = sequence

        self.gru = nn.GRU(obs_size, hid_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hid_size*sequence, act_size)

    @staticmethod
    def pad_sequence(states, episode_end=False, sequence=5, padding_value=0.0):
        r"""Pad a sequence of observations upto sequence size by adding `padding_value` above or below the input.

        It pads above states by default. if episode end is `True` it pads after (see example).

        Args:
            states: tensor of size (batch, sequence_0, features)

            episode_end: bool   

            sequence = 5: int. desired sequence size    

            padding_value = 0.0: float. value to pad tensor with     

        Example:
            >>> a = a = torch.ones(1,2,4)
            >>> print(a) 
            tensor([[[1., 1., 1., 1.],
                 [1., 1., 1., 1.]]])
            >>> pad_sequence(a,episode_end=False, sequence=5)
            tensor([[[0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.]]])
        """
        # assuming batch is always first
        out_dims = (len(states), sequence, states.size(2))
        out_tensor = states[0].data.new(*out_dims).fill_(padding_value)
        length = states.size(1)
        if episode_end:
            out_tensor[:, :length, :] = states
        else:
            out_tensor[:, -length:, :] = states
        return out_tensor

    def forward(self, x, hx=None, episode_end=False):
        r"""Return output and pad the input if observation is less than sequence."""
        if x.size(1) < self.sequence:
            x = self.pad_sequence(x, episode_end, self.sequence)
        out, hx = self.gru(x, hx)
        out = F.relu(out.flatten(1))
        return self.fc(out), hx


class RNNPGII(nn.Module):
    r"""Recurrent plain vanila policy gradient model."""

    def __init__(self, obs_size, act_size, sequence=5, num_layers=2):
        super().__init__()
        self.sequence = sequence
        self.gru = nn.GRU(obs_size, act_size, num_layers, batch_first=True)

    def forward(self, x, hx=None, episode_end=False):
        r"""Return output and pad the input if observation is less than sequence."""
        out, hx = self.gru(x, hx)
        return out, hx


class GRUPG(nn.Module):
    r"""GRU plain vanila policy gradient model."""

    def __init__(self, obs_size, act_size, hid_size=128, num_layers=2):
        super().__init__()
        self.hid_size = hid_size
        self.num_layers = num_layers
        
        self.input = nn.Linear(obs_size, 32)
        self.gru = nn.GRU(32, hid_size, num_layers, batch_first=True)
        self.output = nn.Linear(hid_size, act_size)

    def forward(self, x, hx=None):
        r"""Return output and pad the input if observation is less than sequence."""
        batch_size = x.size(0)
        y = self.input(x)
        y = y.view(batch_size, 1, -1)
        if hx is None:
            hx = torch.zeros((self.num_layers, x.size(0), self.hid_size))
            y, hx = self.gru(y, hx)
        else:
            y, hx = self.gru(y, hx)
        y = self.output(F.relu(y.flatten(1)))
        return y, hx


class RNNAgent(ptan.agent.BaseAgent):
    def __init__(self, model, actions, device='cpu', apply_softmax=True):
        self.model = model
        self.device = device
        self.actions = actions
        self.apply_softmax = apply_softmax
        self.reset()

    def reset(self):
        self.hx = None
        pass

    @torch.no_grad()
    def __call__(self, state, episode_end=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) < 3:
            state.unsqueeze_(0)
        out, self.hx = self.model(state, self.hx, episode_end)
        if self.apply_softmax == True:
            out = torch.softmax(out, dim=1)
        out = out.data.cpu().numpy()[0]
        actions = np.random.choice(self.actions, p=out)
        return np.array(actions), self.hx


class RNNAgentII(ptan.agent.BaseAgent):
    def __init__(self, model, actions, device='cpu', apply_softmax=True):
        self.model = model
        self.device = device
        self.actions = actions
        self.apply_softmax = apply_softmax
        self.reset()

    def reset(self):
        self.hx = None
        pass

    @torch.no_grad()
    def __call__(self, state, episode_end=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) < 3:
            state.unsqueeze_(0)
        out, self.hx = self.model(state, self.hx, episode_end)
        if self.apply_softmax == True:
            out = torch.softmax(out, dim=2)
        out = out.data.cpu().numpy()[-1, -1, :]
        actions = np.random.choice(self.actions, p=out)
        return np.array(actions), self.hx


class GRUAgent(ptan.agent.BaseAgent):
    r"""Do NOT use to collect observations. Incomplete agent, use to play only!"""
    def __init__(self, model, selector=ptan.actions.ArgmaxActionSelector(), device='cpu', apply_softmax=True):
        self.model = model
        self.selector = selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.reset()

    def reset(self):
        self.hx = None
        pass

    @torch.no_grad()
    def __call__(self, state, hx=None):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor([state]).to(self.device)
        # if len(state.shape) < 3:
        #     state.unsqueeze_(1)
        # out, self.hx = self.model(state, hx)
        out = self.model(state)
        if self.apply_softmax == True:
            out = torch.softmax(out, dim=1)
        out = out.data.cpu().numpy()
        actions = self.selector(out)
        return np.array(actions), hx


class BatchGenerator(object):
    r"""
    Summary: generate batchs from environment.

    Parameters
    ----------
    `exp_source` : ptan.experience   

    `train_episodes` : int, optional. Default is None.   

    `params` : SimpleNamespace, optional. Default is None.    

    Returns
    -------
    Itteration returns:
        states
        actions
        discounted_rewards.

    """

    def __init__(self, exp_source, train_episodes: int = None, params: SimpleNamespace = None):
        self.exp_source = exp_source
        self.train_episodes = train_episodes
        self.params = params
        self._total_rewards = []
        self._end_episode_frames = []
        self.__reset__()

    def __reset__(self):
        self.frame = 0
        self.episode = 0

    def pop_last_rewards_frames(self):
        """Return lists of Frames at which epochs ended and their associated undiscounted Rewards."""
        return (self._total_rewards[-self.train_episodes:],
                self._end_episode_frames[-self.train_episodes:])

    def _discount_rewards(self, rewards):
        """
        Summary: calculate the discounted future rewards.

        Takes in list of rewards and discount rate
        Returns the accumlated future values of these rewards
        Example:
        r = [1,1,1,1,1,1]
        gamma = 0.9
        >>> [4.68559, 4.0951, 3.439, 2.71, 1.9, 1.0]
        """
        sum_r = 0.0
        res = []
        for r in reversed(rewards):
            sum_r *= self.params.gamma
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))

    def __iter__(self):
        """
        Iterate over N episodes and return States, Actions and Q values.

        Yields
        ------
        Numpy arraies
            States, Actions, Q_ref_values.

        """
        states, actions, rewards = [], [], []
        disc_rewards = []
        batch_episode = 0
        for exp in self.exp_source:
            self.frame += 1
            new_reward = self.exp_source.pop_total_rewards()
            if new_reward:
                self._total_rewards.append(new_reward[0])
                self._end_episode_frames.append(self.frame)
                disc_rewards.extend(self._discount_rewards(rewards))
                rewards.clear()
                self.episode += 1
                batch_episode += 1
                if batch_episode == self.train_episodes:
                    yield (np.array(states, copy=False),
                           np.array(actions),
                           np.array(disc_rewards))
                    states.clear()
                    actions.clear()
                    disc_rewards.clear()
                    batch_episode = 0
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)


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
