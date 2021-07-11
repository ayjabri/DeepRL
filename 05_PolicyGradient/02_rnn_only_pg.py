#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plain vanilla recurrent policy gradient using RNN without dense layer

Created on Fri Jun 25 14:53:20 2021

@author: ayman
"""

import gym
import ptan
import torch
import argparse
import numpy as np
from lib import model
from time import time
from collections import namedtuple
import torch.nn.functional as F
from datetime import datetime, timedelta
from torchsummary import summary


Experience = namedtuple('Experience',['state','action','reward'])


def preprocess(x):
    return torch.FloatTensor([x])


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
    sum_r = 0.0
    res = []
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


def stack_batch(states, sequence):
    r"""Convert an episode of observations into N sequence."""
    states_v = torch.FloatTensor(np.array(states, copy=False))
    # out_dims = (len(states_v) + sequence - 1, sequence, states_v.size(1))
    out_dims = (len(states_v), sequence, states_v.size(1))
    batch_tensor = states_v.data.new(*out_dims).fill_(0.0)
    for i in range(sequence):
        length = sequence-1-i
        if length == 0:    
            batch_tensor[:,i,:] = states_v
        else:
            batch_tensor[length:,i,:] = states_v[:-length]
    return batch_tensor


@torch.no_grad()
def play(env, agent):
    r"""Play an episode using trained agent."""
    state = env.reset()
    try:
        agent.reset()
    except:
        pass
    rewards = 0
    while True:
        env.render()
        # state_v = torch.FloatTensor([state])
        action = agent(state)[0]
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--rnn', action='store_true', help='Use Recurrent Neural Network')
    # args = parser.parse_args()
    
    ENTROPY_BETA = 0.02
    SEQ = 4
    GAMMA = 0.96
    HID_SIZE = 32
    NUM_LAYERS = 2
    SOLVE = 195
    LR = 1e-3

    env = gym.make('CartPole-v0')
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    
    
    net = model.RNNPGII(obs_size, act_size)
    agent = model.RNNAgentII(net, act_size)

    print(net)
    # print(summary(net, (obs_size,), device='cpu'))
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    total_rewards = []
    print_time = time()
    start = datetime.now()
    frame = 0
    prev_frame = 0
    episode = 0
    episode_rewards = 0
    state = env.reset()
    mean = None
    done = False
    batch = []
    hx = None
    while True:
        while True:
            frame += 1
            action, _ = agent(state, done)
            action = action.item()
            last_state, reward, done, _ = env.step(action)
            episode_rewards += reward
            batch.append(Experience(state, action, reward)) 
            state = last_state
            if done:
                total_rewards.append(episode_rewards)
                episode_rewards = 0
                agent.reset()
                state = env.reset()
                episode += 1
                done = False
                break
        
        mean_reward = np.mean(total_rewards[-100:])
        if time() - print_time > 1:
            speed = (frame - prev_frame)/(time()-print_time)
            prev_frame = frame
            print(
                f"{frame:,}: done {episode} episodes, mean reward {mean_reward:6.3f}", flush=True)
            print_time = time()

        if mean_reward > SOLVE:
            duration = timedelta(seconds=(datetime.now()-start).seconds)
            print(f'Solved in {duration}')
            break

        ### training ###
        states, actions, rewards = list(zip(*batch))
        dis_rewards = discount_rewards(rewards, GAMMA)
        states = np.array(states, copy=False, dtype=np.float32)
        states_v = torch.FloatTensor(states)
        batch_scale_v = torch.FloatTensor(dis_rewards)
        optimizer.zero_grad()
    
        # policy loss
        logits_v, hx = net(states_v.unsqueeze(0))
        # hx = hx.data.cpu()
        # logits_v = net(states_v.unsqueeze(1))[0]
        log_prob_v = F.log_softmax(logits_v, dim=2)[0]
        # log_prob_v = F.log_softmax(logits_v, dim=2)[:,-1,:]
        # Gather probabilities with taken actions
        log_prob_action_v = batch_scale_v * \
            log_prob_v[range(len(actions)), actions]
        policy_loss = - log_prob_action_v.mean()

        # entropy loss
        probs_v = F.softmax(logits_v, dim=2)[0]
        entropy = - (probs_v * log_prob_v).sum(dim=1).mean()
        entropy_loss = - ENTROPY_BETA * entropy
        
        #total loss
        loss = policy_loss + entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        optimizer.step()

        # clear the batch for next episode
        batch.clear()
