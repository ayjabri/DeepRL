#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:30:42 2020

@author: ayman
"""

import gym
import ptan
import time
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from datetime import datetime
import numpy as np
from lib import model, utils, data
from collections import deque



if __name__=='__main__':
    params = data.HYPERPARAMS['pendulum']
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env_id)
        env.seed(params.seed)
        envs.append(env)
    act_net = model.ActorNet(params.obs_size,params.act_size)
    crt_net = model.CriticNet(params.obs_size, params.act_size)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    agent = model.DDPGAgent(act_net, epsilon=0.2)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, params.gamma, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.buffer_size)

    act_optim = torch.optim.Adam(act_net.parameters(), lr=params.lr)
    crt_optim = torch.optim.Adam(crt_net.parameters(),lr=1e-3)

    total_rewards = deque(maxlen=100)
    frame = 0
    last_frame = 0
    episode = 0
    pt = time.time()
    st = datetime.now()
    while True:
        buffer.populate(1)
        frame += 1
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            episode += 1
            total_rewards.append(new_reward[0])
            mean = np.mean(total_rewards)
            if mean > params.bound_solve:
                print(f'Solved in{datetime.now()-st}')
                break
            if time.time()-pt > 1:
                fps = (frame - last_frame) / (time.time()-pt)
                print(f'{frame:7,}: episode:{episode:6}, mean:{mean:7.2f}, speed: {fps:7.2f} fps')
                pt = time.time()
                last_frame = frame
        if len(buffer) < params.init_replay:
            continue

        batch = buffer.sample(params.batch_size)
        states,actions,rewards,dones,last_states=utils.unpack_dqn_batch(batch)
        states_v = torch.tensor(states)
        actions_v = torch.tensor(actions)
        last_states_v = torch.tensor(last_states)
        rewards_v = torch.tensor(rewards)

        # Train Critic
        crt_optim.zero_grad()
        q_sa = crt_net(states_v,actions_v)
        a_last = tgt_act_net.model(last_states_v)
        q_sa_last = tgt_crt_net.model(last_states_v, a_last)
        q_sa_last[dones]=0
        q_ref = rewards_v.unsqueeze(-1) + q_sa_last * params.gamma**params.steps
        critic_loss = F.mse_loss(q_sa,q_ref.detach())
        critic_loss.backward()
        nn_utils.clip_grad_norm_(crt_net.parameters(), max_norm=0.1)
        crt_optim.step()


        # Trian Actor
        act_optim.zero_grad()
        a_curr = act_net(states_v)
        actor_loss = (- crt_net(states_v, a_curr)).mean()
        actor_loss.backward()
        act_optim.step()

        tgt_act_net.alpha_sync(alpha=1-1e-3)
        tgt_crt_net.alpha_sync(alpha=1-1e-3)

