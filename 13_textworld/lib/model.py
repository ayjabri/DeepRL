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
        return Observation(state,cmds)
    
    def reset(self):
        r"""Returns tokenized observation of [State,additional info] + admissible_commands"""
        self.moves = 0
        self.total_rewards = 0
        state, self.last_info = self.env.reset()
        self.admissible_commands = self.last_info["admissible_commands"]
        self.state = self.encode(state, self.last_info)
        return self.state

    def step(self, action:int):
        self.moves += 1
        assert action in range(len(self.admissible_commands))
        self.last_commands = self.admissible_commands[action]
        _obs, reward, done, self.last_info = self.env.step(self.last_commands)
        self.admissible_commands = self.last_info["admissible_commands"]
        if 'intermediate_reward' in self.extra_info:
            reward += self.last_info['intermediate_reward']
        self.state = self.encode(_obs, self.last_info)
        return self.state, reward, done, {}


# class TextWroldDQNAgent(BaseAgent):
#     def __init__(self, model, selector, device, preprocessor):
#         self.model = model
#         self.selector = selector
#         self.device = device
#         self.preprocessor = preprocessor


