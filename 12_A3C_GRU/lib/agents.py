#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 06:34:16 2021

@author: ayman
"""

import torch
import torch.nn.functional as F
import numpy as np


class A3CGruAgent:
    """
    Actor critic agent that returns action It applies softmax by default.
    """

    def __init__(self, model, env, frames, episodes):
        self.model = model
        self.device = model.device
        self.env = env
        self.frames = frames
        self.num_actions = env.action_space.n
        self.episodes = episodes
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.rewards = []

        self.reset()

    def step(self):
        """
        Return actions and hidden states from given list of states.
        
        :param states: list of states
        :return: list of actions
        """
        self.frames.value += 1
        logit, value, self.hx = self.model(self.state, None)
        log_prob = F.log_softmax(logit, dim=1)
        prob = F.softmax(logit, dim=1)
        entropy = -(prob * log_prob).sum(1)
        action = self.select(prob)
        log_prob = log_prob[range(1), action]
        state, self.reward, self.done, _ = self.env.step(action)
        self.state = self.preprocess(state).to(self.device)
        self.reward = min(max(self.reward, -1), 1)
        self.values.append(value)
        self.entropies.append(entropy)
        self.rewards.append(self.reward)
        self.log_probs.append(log_prob)
        if self.done:
            self.episodes.value += 1
            return False
        return True

    @torch.no_grad()
    def play(self, verbose=False):
        """
        Return actions and hidden states from given list of states.
        
        :param states: list of states
        :return: list of actions
        """
        self.clear()
        self.reset()
        while True:
            if verbose: self.env.render()
            logit, _, _ = self.model(self.state)
            prob = F.softmax(logit, dim=1)
            action = self.select(prob)
            state, self.reward, self.done, _ = self.env.step(action)
            self.rewards.append(self.reward)
            if self.done:
                if verbose: print(self.get_total_rewards())
                self.reset()
                break
        self.env.close()


    def preprocess(self, states):
        """Return tensor -> (b,c,h,w).(device)."""
        np_states = np.expand_dims(states, 0)
        return torch.tensor(np_states)

    def select(self, prob):
        """Select from a probability distribution."""
        # return np.random.choice(self.num_actions, p=prob.data.cpu().numpy()[0])
        return prob.multinomial(1).data.cpu().numpy()[0]

    def get_total_rewards(self):
        return sum(self.rewards)

    def clear(self):
        """Use to clear all values. Use with reset if you want clean start."""
        self.values.clear()
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()
        if self.hx is not None:
            self.hx = self.hx.detach()

    def reset(self):
        """Reset the agent. Use when episode is done but want to continue training."""
        self.hx = None
        self.reward = 0
        self.done = False
        self.state = self.preprocess(self.env.reset()).to(self.device)
