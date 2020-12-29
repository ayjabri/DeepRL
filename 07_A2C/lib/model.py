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
        return (self.policy(y),self.value(y))


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


class BatchGenerator:
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

    def __init__(self, exp_source, params:SimpleNamespace, batch_size:int=256, baseline=False):
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

    def calc_baseline(self, rewards):
        """Reduce rewards by their at point averages if baseline is True."""
        if not self.baseline: return rewards
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
        states, actions, rewards,dones,last_states = [], [], [], [],[]
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
            states.append(np.array(exp.state,copy=False))
            actions.append(int(exp.action))
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            last_states.append(np.array(exp.last_state, copy=False) if\
                               exp.last_state is not None else np.array(exp.state,copy=False))



class EpisodeGenerator:
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

    def __init__(self, exp_source, params:SimpleNamespace, n_episodes:int=10, baseline=False):
        self.exp_source = exp_source
        self.params = params
        self.n_episodes = n_episodes
        self.baseline = baseline
        self._total_rewards = []
        self._end_episode_frames = []
        self.__reset__()

    def __reset__(self):
        """Set frame and episode counters to 0."""
        self.frame = 0
        self.episode = 0

    def calc_baseline(self, rewards):
        """Reduce rewards by their averages if baseline is True."""
        if not self.baseline: return rewards
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
        states, actions, rewards,dones,last_states = [],[],[],[],[]
        batch_episode = 0
        for exp in self.exp_source:
            self.frame += 1
            new_reward = self.exp_source.pop_total_rewards()
            if new_reward:
                self._total_rewards.append(new_reward[0])
                self._end_episode_frames.append(self.frame)
                self.episode += 1
                batch_episode += 1
            if batch_episode >= self.n_episodes:
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
                batch_episode = 0
            states.append(np.array(exp.state,copy=False))
            actions.append(int(exp.action))
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            last_states.append(np.array(exp.last_state, copy=False) if\
                               exp.last_state is not None else np.array(exp.state,copy=False))



class CategoricalSelector(ptan.actions.ActionSelector):
    """Sample Actions from Categorical distribution."""

    def __call__(self,prob):
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
        distribution = torch.distributions.Categorical(prob, validate_args=True)
        actions = distribution.sample()
        return actions.numpy()


@torch.no_grad()
def unpack_a2c_batch(batch,crt_net,params):
    """
    Definition: returns states and actions plut advantage value: Adv(s,a) = Q(s,a) - V(s`).

    Parameters
    ----------
    batch : TYPE
        DESCRIPTION.
    act_net : TYPE
        DESCRIPTION.
    crt_net : TYPE
        DESCRIPTION.
    gamma : TYPE, optional
        DESCRIPTION. The default is 0.99.
    steps : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    states : Tensor
        DESCRIPTION.
    actions : Tensor
        DESCRIPTION.
    Q(s,a) : Tensor
        Q(s,a) = r + gamma ^ steps * V(s`)
        V(s`) = critic_net(s`) --- Only for not_dones last states

    """
    states, actions, rewards, dones, last_states = batch
    not_dones = dones == False # Switch the dones to no_dones array
    if not_dones.any():
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False))
        last_values_np = crt_net(last_states_v[not_dones]).data.numpy()
        rewards[not_dones] += last_values_np[:,0] * params.gamma**params.steps
    q_sa_v = np.array(rewards,copy=False,dtype=np.float32)
    return states, actions, q_sa_v



