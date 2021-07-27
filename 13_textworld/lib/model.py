#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:39:57 2021

@author: ayman
"""
from ptan.agent import BaseAgent
from gym import Wrapper
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np


Observation = namedtuple("Observation", ["obs", "admissible_commands"])


class TextWrapper(Wrapper):
    r"""Wrap TextWorld environment such that it returns consistant observations upon reset and step functions"""

    def __init__(self, env, trainable_extra_info=("inventory", "description")):
        r"""
        Select from a list of extra info to include in training. These attributes should've been 
        requested as extra information when creating the environment.
        env: worldtext environment 
        extra_info: tuple of additional information to encode as part of training
        """
        super(TextWrapper, self).__init__(env=env)
        self.trainable_extra_info = trainable_extra_info
        self.tokenize_cmd = env.action_space.tokenize
        self.tokenize_obs = env.observation_space.tokenize
        self.last_commands = []
        self.admissible_commands = []
        self._init_obs()

    def _init_obs(self):
        self.extra_info = []
        for att in dir(self.env.request_infos):
            value = getattr(self.env.request_infos, att)
            if isinstance(value, bool) and value:
                self.extra_info.append(att)
        for ext in self.trainable_extra_info:
            if ext not in self.extra_info:
                raise ValueError(
                    f"{ext} is not in the environment extra information")
        if 'admissible_commands' not in self.extra_info:
            raise ValueError("Need to have admissible commnads in this model")
        pass

    def encode(self, obs, info):
        state = [self.tokenize_obs(obs)]
        cmds = list(map(self.tokenize_cmd, info["admissible_commands"]))
        for rec in self.trainable_extra_info:
            state.append(self.tokenize_obs(info[rec]))
        return Observation(state, cmds)

    def reset(self):
        r"""Returns tokenized observation of [State,additional info] + admissible_commands"""
        self.moves = 0
        self.total_rewards = 0
        state, self.last_info = self.env.reset()
        self.admissible_commands = self.last_info["admissible_commands"]
        self.state = self.encode(state, self.last_info)
        return self.state

    def step(self, action: int):
        self.moves += 1
        assert action in range(len(self.admissible_commands))
        self.last_commands = self.admissible_commands[action]
        _obs, reward, done, self.last_info = self.env.step(self.last_commands)
        self.admissible_commands = self.last_info["admissible_commands"]
        if 'intermediate_reward' in self.extra_info:
            reward += self.last_info['intermediate_reward']
        self.state = self.encode(_obs, self.last_info)
        return self.state, reward, done, {}


class TextWorldAgent(BaseAgent):
    def __init__(self, model, preprocessor, device='cpu', eps_final=0.2, eps_frame=20_000):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self._epsilon = 1.0
        self.eps_final = eps_final
        self.step = (1.0-eps_final)/eps_frame

    @torch.no_grad()
    def __call__(self, state, agent_state=None):
        obs, commands = self.preprocessor.prep(state)
        actions = self.model.q_vals(obs,commands).argmax().data.cpu().numpy()
        if np.random.random() < self.epsilon:
            actions = np.random.choice(range(len(commands)))
            return np.array(actions), agent_state
        return actions, agent_state

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def frame(self, value):
        self._epsilon = max(self.eps_final, 1-self.step*value)
        


class DQNet(nn.Module):
    def __init__(self, obs_size, cmd_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size + cmd_size, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, obs, cmd):
        x = torch.cat((obs, cmd), dim=1)
        return self.net(x).squeeze(0)

    def q_vals(self, obs, commands: list):
        for cmds in commands:
            q_val = torch.cat([self(obs, c) for c in cmds])
        return q_val
