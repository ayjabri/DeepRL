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


class PGNet(nn.Module):
    r"""Plain vanilla policy gradient network."""

    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hid_size)
        self.output = nn.Linear(hid_size, act_size, bias=True)

    def forward(self, x):
        """Feed forward."""
        y = F.relu(self.fc1(x))
        return self.output(y)


class BatchGenerator(object):
    def __init__(self, exp_source, params: SimpleNamespace, batch_size: int = 256, baseline=False):
        assert (exp_source.steps > 1)
        self.exp_source = exp_source
        self.params = params
        self.batch_size = batch_size
        self.baseline = baseline
        self._total_rewards = []
        self._end_episode_frames = []
        self.__reset__()

    def __reset__(self):
        """Set frame and episode counters to 0."""
        self.frame = 0
        self.episode = 0

    def pop_last_rewards_frames(self):
        """Return lists of Frames at which epochs ended and their associated undiscounted Rewards."""
        return (self._total_rewards[-self.train_episodes:],
                self._end_episode_frames[-self.train_episodes:])

    def calc_baseline(self, rewards):
        """Reduce rewards by their at point averages if baseline is True."""
        if not self.baseline:
            return rewards
        baseline = np.cumsum(rewards)/np.arange(1, self.batch_size + 1)
        return rewards - baseline

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
                       self.calc_baseline(rewards))
                states.clear()
                actions.clear()
                disc_rewards.clear()
                rewards.clear()
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
