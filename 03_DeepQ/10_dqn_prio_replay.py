#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:19:17 2020

@author: Ayman Al Jabri
Solves Mountain Car very easily in 357 episodes in less than 5:00 minutes!
"""

import torch
import gym
import ptan
from lib import model, utils, hyperparameters
import numpy as np
from tensorboardX import SummaryWriter
from collections import Counter

if __name__=='__main__':
    wrap = True
    BETA_FRAME = 50_000
    BETA_START = 0.1
    params = hyperparameters.HYPERPARAMS['mountaincar']

    class AdvancedReward(gym.RewardWrapper):
        def __init__(self, env):
            super().__init__(env=env)
            self.env = env
            self.count = Counter()
        def reward(self, reward):
            sx = np.round(self.state[0],2)
            if sx == 0.5:
                return 100
            self.count[sx] += 1
            reward += np.exp((1/self.count[sx]))
            return reward


    def wrap_em(env):
        env = ptan.common.wrappers.NoopResetEnv(env)
        env = ptan.common.wrappers.FireResetEnv(env)
        env = ptan.common.wrappers.EpisodicLifeEnv(env)
        return env


    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env_id)
        if wrap:
            env = AdvancedReward(env)
        env.seed(124)
        envs.append(env)

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n


    support = np.linspace(params.vmin, params.vmax, params.natoms)
    dz = (params.vmax-params.vmin)/(params.natoms-1)

    net = model.C51Net(params.obs_size, params.act_size, params.vmin, params.vmax, params.natoms)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x) , selector,
                                preprocessor=ptan.agent.float32_preprocessor)
    eps_tracker = ptan.actions.EpsilonTracker(selector,params.eps_start,params.eps_end, params.eps_frames)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma, steps_count=params.steps)
    # buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.buffer_size)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source,params.buffer_size, alpha=0.6)
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(net.parameters(), params.lr)

    frame_idx = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame_idx += params.n_envs
            BETA = min(1.,BETA_START + frame_idx/BETA_FRAME)
            buffer.populate(params.n_envs)
            eps_tracker.frame(frame_idx)
            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                mean = tracker.reward(new_reward[0], frame_idx, epsilon=selector.epsilon)
                if mean and mean > params.bound_solve:
                    print('Solved')
                    break
            if len(buffer) < params.init_replay:
                continue

            optimizer.zero_grad()
            batch,batch_indicies,weights = buffer.sample(params.batch_size * params.n_envs, BETA)
            loss,prios = utils.calc_prio_dist_loss(batch, weights, net, tgt_net,
                                          params.gamma, params.vmin, params.vmax, params.natoms, dz)
            loss.backward()
            optimizer.step()
            buffer.update_priorities(batch_indicies, prios)

            if frame_idx % params.sync_net == 0:
                tgt_net.sync()
