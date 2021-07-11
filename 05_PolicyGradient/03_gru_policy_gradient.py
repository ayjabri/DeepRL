#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:39:33 2021

@author: ayman
"""
import argparse
from time import time
from collections import namedtuple
from itertools import count
from datetime import datetime, timedelta

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


Experience = namedtuple('Experience', ['state', 'action', 'reward'])


class GRULayer(nn.Module):
    r"""GRU plain vanila policy gradient model."""

    def __init__(self, obs_size, act_size, hid_size=128, num_layers=2):
        super().__init__()
        self.hid_size = hid_size
        self.num_layers = num_layers

        # self.input = nn.Linear(obs_size, 32)
        self.gru = nn.GRU(obs_size, hid_size, num_layers, batch_first=True)
        self.output = nn.Linear(hid_size, act_size)

    def forward(self, obs, hidden_state=None):
        r"""Return output and pad the input if observation is less than sequence."""
        batch_size = obs.size(0)
        # y = self.input(x)
        obs = obs.view(batch_size, 1, -1)
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_layers, batch_size, self.hid_size))
        obs, hidden_state = self.gru(obs, hidden_state)
        obs = self.output(F.relu(obs.flatten(1)))
        return obs, hidden_state


class GRUCell(nn.Module):
    r"""Plain vanilla policy gradient network."""

    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()
        self.fc1 = nn.GRUCell(obs_size, hid_size)
        self.output = nn.Linear(hid_size, act_size, bias=True)

    def forward(self, obs):
        """Feed forward."""
        output = F.relu(self.fc1(obs))
        return self.output(output)


def discount_rewards(rewards, gamma):
    r"""
    Summary: calculate the discounted future rewards.

    Takes in list of rewards and discount rate
    Returns the accumlated future values of these rewards

    Example:
        >>> r = [1,1,1,1,1,1]
        >>> gamma = 0.9
        >>> [4.68559, 4.0951, 3.439, 2.71, 1.9, 1.0]
    """
    sum_rewards = 0.0
    res = []
    for rew in reversed(rewards):
        sum_rewards *= gamma
        sum_rewards += rew
        res.append(sum_rewards)
    return list(reversed(res))


@torch.no_grad()
def generate_eipsodes(env, gamma, cell, num_episodes=2):
    r"""Generate n episode observations:(state,action,discounted_rewards,total_rewards,frames)"""
    episode = 0
    batch_total_rewards = 0
    hid_sc = None
    act_size = env.action_space.n
    states, actions, rewards = [], [], []
    dis_r = []
    state = env.reset()
    for frame in count():
        state_v = torch.FloatTensor([state])
        # if np.random.random() > 0.5: # placeholder to injects noise
        #     state_v.fill_(0.0)
        #     pass
        if cell:
            prob = net(state_v)  # Linear
        else:
            prob, hid_sc = net(state_v, hid_sc)  # GRU
        prob = torch.softmax(prob, dim=1).data.cpu().numpy()[0]
        action = np.random.choice(act_size, p=prob)
        last_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        if done:
            dis_r.extend(discount_rewards(rewards, gamma))
            batch_total_rewards += sum(rewards)
            rewards.clear()
            episode += 1
            state = env.reset()
            hid_sc = None
            if episode == num_episodes:
                yield (np.array(states, copy=False),
                       np.array(actions), np.array(dis_r),
                       batch_total_rewards/num_episodes,
                       frame)
                states.clear()
                actions.clear()
                rewards.clear()
                dis_r.clear()
                batch_total_rewards = 0
                episode = 0
        state = last_state


@torch.no_grad()
def play(env, net, cell):
    r"""Play an episode using trained agent."""
    state = env.reset()
    rewards = 0
    hid_sc = None
    while True:
        env.render()
        state_v = torch.FloatTensor([state])
        if not cell:
            prob, hid_sc = net(state_v, hid_sc)
        else:
            prob = net(state_v)
        action = prob.softmax(dim=1).argmax(1).numpy()
        state, rew, done, _ = env.step(action.item())
        rewards += rew
        if done:
            print(rewards)
            break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', action='store_true',
                        help='Use GRUCell instead of GRU layer')
    args = parser.parse_args()

    ENTROPY_BETA = 0.02
    GAMMA = 0.99
    HID_SIZE = 64
    NUM_LAYERS = 1
    SOLVE = 195
    LR = 0.01
    N_EPS = 2
    SEED = 155

    env = gym.make('CartPole-v0')  # 'LunarLander-v2'
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    if args.cell:
        net = GRUCell(obs_size, act_size, HID_SIZE)  # Linear
    else:
        net = GRULayer(obs_size, act_size, HID_SIZE, NUM_LAYERS)  # GRU

    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    total_rewards = []
    print_time = time()
    start = datetime.now()
    FRAME = 0
    PRV_FRAME = 0
    for episode, batch in enumerate(generate_eipsodes(env, GAMMA, args.cell, num_episodes=N_EPS)):
        states, actions, rewards, batch_total_rewards, FRAME = batch
        total_rewards.append(batch_total_rewards)
        mean_reward = np.mean(total_rewards[-100:])
        if time() - print_time > 1:
            speed = (FRAME - PRV_FRAME)/(time()-print_time)
            PRV_FRAME = FRAME
            print(
                f"{FRAME:,}: done {episode} episodes, mean reward {mean_reward:6.3f}, speed:{speed:.0f} fps", flush=True)
            print_time = time()

        if mean_reward > SOLVE:
            duration = timedelta(seconds=(datetime.now()-start).seconds)
            print(f'Solved in {duration}')
            break

        ### training ###
        states_v = torch.FloatTensor(states)
        batch_scale_v = torch.FloatTensor(discount_rewards(rewards, GAMMA))
        actions_v = torch.LongTensor(actions)
        optimizer.zero_grad()

        # policy loss
        if args.cell:
            logit = net(states_v)  # Linear
        else:
            logit, hn = net(states_v)  # GRU
        log_p = F.log_softmax(logit, dim=1)

        # Gather probabilities with taken actions
        log_p_a = batch_scale_v * log_p[range(len(actions)), actions]
        policy_loss = - log_p_a.mean()

        # entropy loss
        probs_v = F.softmax(logit, dim=1)
        entropy = - (probs_v * log_p).sum(dim=1).mean()
        entropy_loss = - ENTROPY_BETA * entropy

        # total loss
        loss = policy_loss + entropy_loss
        loss.backward()
        optimizer.step()
