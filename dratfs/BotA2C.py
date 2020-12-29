#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:00:09 2020
@author: aymanjabri

Actor_Critic implementation in training a 4 legged robot with 8 degrees of freedom
The bot has 4 legs, each has two motors
Policy Loss: ∇𝐽 = ∇𝜃 log 𝜋𝜃(𝑎|𝑠) (𝑅 − 𝑉𝜃(𝑠))
    Where: log 𝜋𝜃(𝑎|𝑠) = −((𝑥 − 𝜇)^2 / (2𝜎^2)) − log √2𝜋𝜎^2
Entropy Loss: 𝐿𝐻 = 𝜋𝜃(𝑠) log 𝜋𝜃(𝑠)
Value Loss: Mean_square error beteween the value of the network and the one estimated by Bellman equation

"""


import gym
import pybullet_envs
import torch
import torch.nn as nn
import ptan
import numpy as np
import math.pi


# =============================================================================
# Create the A2C network with three heads: mu, var and value
# =============================================================================
class A2C(nn.Module):
    def __init__(self, obs_size, act_size, hid_size):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(obs_size, hid_size),
                                   nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(hid_size, act_size),
                                nn.Tanh())
        self.var = nn.Sequential(nn.Linear(hid_size, act_size),
                                 nn.Softplus())
        self.value = nn.Linear(hid_size, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)

# =============================================================================
# The agent is based on ptan.BaseAgent. It takes observations & returns actions
# =============================================================================
class A2CAgent(ptan.agent.BaseAgent):
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v,var_v, _ = self.model(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


# =============================================================================
# Test function is the critic part. It plays n number of games and returns
# Rewards and steps moved
# =============================================================================
def test(net, env, count=10, device='cpu'):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor(obs)
            mu_v = net(obs_v)[0]
            action = mu_v.data.cpu().numpy().squeeze(0)
            action = np.clip(action, -1, 1)
            obs, reward, done,_ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards/count, steps/count


# =============================================================================
# Calculate the log probablity of Gaussian distribution
# =============================================================================
def calc_logprob(mu_v, var_v, action_v):
    p1 = -((action_v - mu_v)**2)/(2*var_v.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2*math.pi*var_v))
    return p1+p2


if __name__=='__main__':
    ENV_ID = 'MinitaurBulletEnv-v3'
    GAMMA = 0.99
    REWARDS_STEPS = 2
    BATCH_SIZE = 32
    LEARNING_RATE  = 1e-3
    ENTROPY_BETA = 1e-4
    TEST_ITERS = 1000
