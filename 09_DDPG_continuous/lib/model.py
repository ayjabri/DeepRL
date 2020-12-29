#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 07:06:18 2020

@author: ayman
"""
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class ActorNet(nn.Module):
    """Deep Deterministic Policy Gradient Actor Network."""

    def __init__(self,obs_size, act_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.actions = nn.Linear(64, act_size)
        self.activation = nn.Tanh()

    def forward(self,x):
        """Feed forward function."""
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        return self.activation(self.actions(y))


class CriticNet(nn.Module):
    """Deep Deterministic Policy Gradient Actor Network."""

    def __init__(self,obs_size,act_size):
        super().__init__()
        self.base = nn.Linear(obs_size, 256)
        self.q1 = nn.Linear(256+act_size, 128)
        self.q2 = nn.Linear(128, 1)

    def forward(self,s,a):
        """Feed forward function."""
        base = F.relu(self.base(s))
        y = torch.cat([base,a],dim=1)
        y = F.relu(self.q1(y))
        return self.q2(y)


class DDPGAgent(ptan.agent.BaseAgent):
    """Deep Deterministic Policy Gradient Actor Network."""

    def __init__(self, act_net, preprocessor= None, epsilon=0.3):
        self.model = act_net
        self.epsilon = epsilon
        if preprocessor is None:
            self.preprocessor = ptan.agent.float32_preprocessor
        else:
            self.preprocessor = preprocessor

    def __call__(self, state, agent_states=None):
        """
        Return actions from actor network after adding a noise from normal distribution.

        The Noise is scaled by epsilon and clipped to range [-1,1]
        """
        state_v = self.preprocessor(state)
        actions_np = self.model(state_v).data.numpy()
        noise_np = np.random.normal(size=actions_np.shape)
        actions_np += noise_np * self.epsilon
        return  (np.clip(actions_np, -1, 1), agent_states)



def unpack_dqn_batch(batch):
    """
    Definition: unpack_dqn_batch(batch)

    Unpack a batch of observations

    Parameters
    ----------
    batch : a list contains a namedtuples of (state,action,reward,last_state)

    Returns
    -------
    states:float32

    actions:int

    rewards:float64

    dones:bool

    last_states:float32

    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    return (np.array(states, copy=False, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(last_states, copy=False, dtype=np.float32))