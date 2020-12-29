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
    r"""
    Summary: generate batchs from environment.

    Parameters
    ----------
    exp_source : ptan.experience
         .
    train_episodes : int, optional
         . The default is None.
    params : SimpleNamespace, optional
         . The default is None.

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
