# -*- coding: utf-8 -*-

from ptan.actions import ActionSelector
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import numpy as np


class RLNet(nn.Module):
    """
    Simple Reinforcement Learning Network with two layers only
    """

    def __init__(self, obs_size, act_size):
        super(RLNet, self).__init__()
        self.obs_size = obs_size
        self.act_size = act_size
        self.base = nn.Linear(obs_size, 128)
        self.out = nn.Linear(128, act_size)

    def forward(self, x):
        x = F.relu(self.base(x))
        return self.out(x)


class BatchGenerator:
    """

    Parameters
    ----------
    exp_source : ptan.experience
        DESCRIPTION.
    train_episodes : int, optional
        DESCRIPTION. The default is None.
    params : SimpleNamespace, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Itteration returns:
        states
        actions
        discounted_rewards.

    """

    def __init__(self, exp_source, train_episodes: int = None, params: SimpleNamespace = None):
        self.exp_source = exp_source
        self.batch_size = params.batch_size
        self.train_episodes = train_episodes
        self.params = params
        self.train_episodes = train_episodes
        self._total_rewards = []
        self._end_episode_frames = []
        self.reset()

    def reset(self):
        self.frame = 0
        self.episodes = 0
        self.new_reward = 0.0

    def batch_mean(self):
        batch_rewards = list(self._total_rewards)[-self.train_episodes:]
        return np.mean(batch_rewards)

    def discount_rewards(self, rewards):
        '''
        Function to calculate the discounted future rewards
        Takes in list of rewards and discount rate
        Returns the accumlated future values of these rewards
        Example:
        r = [1,1,1,1,1,1]
        gamma = 0.9
        >>> [4.68559, 4.0951, 3.439, 2.71, 1.9, 1.0]
        '''
        sum_r = 0.0
        res = []
        for r in reversed(rewards):
            sum_r *= self.params.gamma
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))

    def __iter__(self):
        states, actions, rewards = [], [], []
        disc_rewards = []
        episode = 0
        for exp in self.exp_source:
            self.frame += 1
            new_reward = self.exp_source.pop_total_rewards()
            if new_reward:
                self._total_rewards.append(new_reward[0])
                self._end_episode_frames.append(self.frame)
                disc_rewards.extend(self.discount_rewards(rewards))
                rewards.clear()
                episode += 1
                self.episodes += 1
                if episode == self.train_episodes:
                    yield (np.array(states, copy=False),
                           np.array(actions),
                           np.array(disc_rewards))
                    states.clear()
                    actions.clear()
                    disc_rewards.clear()
                    episode = 0
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)


class CategoricalSelector(ActionSelector):
    """
    Select from Categorical distribution using pytorch
    """

    def __call__(self, prob):
        if isinstance(prob, np.ndarray):
            prob = torch.Tensor(prob)
        assert isinstance(prob, torch.Tensor)
        distribution = torch.distributions.Categorical(
            prob, validate_args=True)
        actions = distribution.sample()
        return actions.numpy()
