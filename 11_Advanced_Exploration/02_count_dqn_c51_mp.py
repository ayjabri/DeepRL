#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 07:04:13 2020

@author: ayman
"""
import os
import torch
import gym
import ptan
from lib import model, data
import numpy as np
from tensorboardX import SummaryWriter
from collections import Counter
import torch.multiprocessing as mp
import torch.nn.functional as F
from datetime import datetime, timedelta

wrap = False

params = data.HYPERPARAMS['lander1']

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


# def wrap_em(env):
#     env = ptan.common.wrappers.NoopResetEnv(env)
#     env = ptan.common.wrappers.FireResetEnv(env)
#     env = ptan.common.wrappers.EpisodicLifeEnv(env)
#     return env

def unpack_dqn_batch(batch):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
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


def calc_proj_dist(prob,rewards,dones,vmin,vmax,natoms,dz,gamma):
    proj = np.zeros(prob.shape)
    for atom in range(natoms):
        v = rewards + (vmin + atom * dz) * gamma
        v = np.maximum(vmin,np.minimum(v,vmax))
        idx = (v-vmin)/dz
        l = np.floor(idx).astype(int)
        u = np.ceil(idx).astype(int)
        eq_mask = l==u
        proj[eq_mask,l[eq_mask]] += prob[eq_mask,atom]
        neq_mask = l!=u
        proj[neq_mask,l[neq_mask]] += prob[neq_mask,atom] * (idx - l)[neq_mask]
        proj[neq_mask,l[neq_mask]] += prob[neq_mask,atom] * (u - idx)[neq_mask]
    if dones.any():
        proj[dones] = 0.
        d_v = np.maximum(vmin,np.minimum(rewards[dones],vmax))
        d_idx = (d_v-vmin)/dz
        proj[dones,d_idx.astype(int)] = 1.0
    return proj



def c51_calc_func(tgt_net,c51_queue,buffer,params,dz):
    while True:
        batch = buffer.sample(params.batch_size * params.n_envs)
        states, actions, rewards, dones, last_states = unpack_dqn_batch(batch)
        # states_v = torch.FloatTensor(states)
        actions_v = torch.tensor(actions)
        last_states_v = torch.FloatTensor(last_states)

        next_dist, next_qvals = tgt_net.target_model.both(last_states_v)
        next_acts = next_qvals.max(dim=1)[1].data.numpy() #next actions
        next_probs = tgt_net.target_model.softmax(next_dist).data.numpy()
        next_probs_actions = next_probs[range(len(next_acts)), next_acts]

        proj_dist = calc_proj_dist(next_probs_actions, rewards, dones,
                  params.vmin, params.vmax, params.natoms, dz, params.gamma)
        # proj_dist_v = torch.FloatTensor(proj_dist)
        c51_queue.put((states,actions,proj_dist))


def mp_c51_loss(net,states_v,actions_v,proj_dist_v):
    '''
    Calculate the loss of a categorical DQN batch with C51*N_actions size
    '''
    distr_v = net(states_v)
    sa_values = distr_v[range(len(actions_v)), actions_v.data]
    log_sa_values = F.log_softmax(sa_values, dim=1)
    loss = -log_sa_values * proj_dist_v
    return loss.sum(dim=1).mean()


if __name__=='__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"

    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env_id)
        if wrap:
            env = AdvancedReward(env)
        env.seed(params.seed)
        envs.append(env)

    dz = (params.vmax-params.vmin)/(params.natoms-1)

    net = model.C51Net(params.obs_size, params.act_size, params.vmin, params.vmax, params.natoms)
    net.share_memory()
    tgt_net = ptan.agent.TargetNet(net)
    tgt_net.target_model.share_memory()
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x) , selector,
                                preprocessor=ptan.agent.float32_preprocessor)
    eps_tracker = ptan.actions.EpsilonTracker(selector,params.eps_start,params.eps_end, params.eps_frames)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma, steps_count=params.steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.buffer_size)
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(net.parameters(), params.lr)

    c51_queue = mp.Queue(1)
    proc = mp.Process(target=c51_calc_func, args=(tgt_net,c51_queue,buffer,params,dz))

    st = datetime.now()
    frame_idx = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame_idx += params.n_envs
            buffer.populate(params.n_envs)
            eps_tracker.frame(frame_idx)
            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                mean = tracker.reward(new_reward, frame_idx, epsilon=selector.epsilon)
                if mean and mean > params.bound_solve:
                    delta = timedelta(seconds=(datetime.now()-st).seconds)
                    print('Solved within {delta}')
                    break
            if len(buffer) < params.init_replay:
                continue
            elif len(buffer) == params.init_replay:
                proc.start()

            optimizer.zero_grad()
            s,a,d = c51_queue.get()
            states_v = torch.FloatTensor(s)
            actions_v = torch.LongTensor(a)
            proj_dist_v = torch.FloatTensor(d)

            distr_v = net(states_v)
            sa_values = distr_v[range(len(actions_v)), actions_v.data]
            log_sa_values = F.log_softmax(sa_values, dim=1)
            loss = -log_sa_values * proj_dist_v
            loss = loss.sum(dim=1).mean()
            loss.backward()
            optimizer.step()

            del s,a,d
            if frame_idx % params.sync_net == 0:
                tgt_net.sync()

    proc.terminate()
    proc.join()
