#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:19:17 2020

@author: ayman
"""

import torch
import gym
import ptan
from lib import model, utils, wrappers
import numpy as np
from tensorboardX import SummaryWriter

env_id = 'Bowling-ramNoFrameskip-v4'
wrap = False
vmin = 0
vmax = 60
natoms = 51
gamma = 0.99
buffer_size = 50_000
steps = 2
n_envs = 4
lr = 1e-3
solve = 60
init_replay = 5000


def wrap_em(env):
    env = ptan.common.wrappers.NoopResetEnv(env)
    env = ptan.common.wrappers.FireResetEnv(env)
    env = ptan.common.wrappers.EpisodicLifeEnv(env)
    return env


envs = []
for _ in range(n_envs):
    env = wrappers.MaxAndSkipEnv(gym.make(env_id), 4)
    if wrap:
        env = wrap_em(env)
    env.seed(124)
    envs.append(env)

obs_size = env.observation_space.shape[0]
act_size = env.action_space.n


support = np.linspace(vmin, vmax, natoms)
dz = (vmax-vmin)/(natoms-1)

net = model.C51Net(obs_size, act_size, vmin, vmax, natoms)
tgt_net = ptan.agent.TargetNet(net)
selector = ptan.actions.ArgmaxActionSelector()
agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector,
                            preprocessor=ptan.agent.float32_preprocessor)
exp_source = ptan.experience.ExperienceSourceFirstLast(
    envs, agent, gamma, steps_count=steps)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size)
writer = SummaryWriter()
optimizer = torch.optim.Adam(net.parameters(), lr)

frame_idx = 0
with ptan.common.utils.RewardTracker(writer) as tracker:
    while True:
        frame_idx += n_envs
        buffer.populate(n_envs)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            mean = tracker.reward(new_reward, frame_idx)
            if mean and mean > solve:
                print('Solved')
                break
        if len(buffer) < init_replay:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(64 * n_envs)
        loss = utils.calc_dist_loss(batch, net, tgt_net, gamma**steps,
                                    vmin, vmax, natoms, dz)
        loss.backward()
        optimizer.step()

        if frame_idx % 1000 == 0:
            tgt_net.sync()