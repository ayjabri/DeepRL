# -*- coding: utf-8 -*-
'''
Run this file to solve one of the available games using policy gradient with and without baseline!
To play without baseline set the steps to 1 (it is a must!)
If steps are > 1: the script will automatically set baseline to True and reduce rewards by the
their averages

Now adays nobody uses plain_vanilla policy gradient but this is for illustration purpose.
Policy gradient: is an improvment of 'Reinforce' method. It addresses two important problems:
a. high gradient variance: it normalizes Q(s,a) value by substracting a baseline from it (the baseline itself
    can be calculated using different methods; average_rewards, moving average or state value. Here we are using average rewards
b. exploration: it improves exploration by punishing the agent when it is too certain of the action. it does so
    by simply substracting entropy value from policy loss

Steps:
1- Initialize the network with random weights.
2- Play N full episodes, saving their (s,a,r,s') transitions
3- For every step t of every episode, calculate the total discounted 
    rewards of subsequent steps 𝑄(𝑘,𝑡) = Σ 𝛾^𝑖 * 𝑟_𝑖 - baseline
4- Calculate policy loss = ℒ = −Σ 𝑄(𝑘,𝑡) log 𝜋(𝑠,𝑎)
5- Perform SGD update of weights
6- Repeat from step 2

Usually solves in 0:00:10 after playing 67 epochs when using the same paramteres 
in this file it should solve in about 8 seconds. just make sure to use ADAM optimizer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import gym
import ptan
from datetime import datetime, timedelta
from tensorboardX import SummaryWriter
from time import time
from collections import deque


games = {
        'cartpole':{    'ENV_ID':'CartPole-v0',
                        'GAMMA': 0.95,
                        'LR': 1e-2,
                        'ENTROPY_BETA': 0.01,
                        'PLAY_EPISODES': 10,
                        'BOUND': 195,
                        },
        'lander':{ 'ENV_ID':'LunarLander-v2',
                        'GAMMA': 0.99,
                        'LR': 1e-3,
                        'ENTROPY_BETA': 0.04,
                        'PLAY_EPISODES': 10,
                        'BOUND': 150,
                        },
        'cartpole500':{ 'ENV_ID':'CartPole-v1',
                        'GAMMA': 0.99,
                        'LR': 1e-2,
                        'ENTROPY_BETA': 0.01,
                        'PLAY_EPISODES': 10,
                        'BOUND': 450,
                        },
        'freeway':{ 'ENV_ID':'Freeway-ramNoFrameskip-v4',
                        'GAMMA': 0.99,
                        'LR': 1e-3,
                        'ENTROPY_BETA': 0.02,
                        'PLAY_EPISODES': 4,
                        'BOUND': 20,
                        },
        'breakout':{ 'ENV_ID':'Breakout-ram-v4',
                        'GAMMA': 0.99,
                        'LR': 1e-3,
                        'ENTROPY_BETA': 0.02,
                        'PLAY_EPISODES': 4,
                        'BOUND': 200,
                        },
        'bowling':{ 'ENV_ID':'Bowling-ram-v4',
                        'GAMMA': 0.99,
                        'LR': 1e-3,
                        'ENTROPY_BETA': 0.01,
                        'PLAY_EPISODES': 4,
                        'BOUND': 60,
                        },
}



class Net(nn.Module):
    '''
    Simple neural network with two linear layers with one ReLU activation. Nothing fancy!
    '''
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(obs_shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, n_actions))
    
    def forward(self, x):
        return self.layer(x)


def calc_q_values(rewards, baseline=True):
    if not baseline: return rewards
    baseline = np.cumsum(rewards)/np.arange(1, len(rewards)+1)
    return rewards - baseline


def discount_rewards(rewards, gamma):
    '''
    Function to calculate the discounted future rewards
    Takes in list of rewards and discount rate
    Returns the accumlated future values of these rewards
    Example:
    r = [1,1,1,1,1,1]
    gamma = 0.9
    >>> [4.68559, 4.0951, 3.439, 2.71, 1.9, 1.0]
    '''
    res = 0
    l = []
    for i in reversed(rewards):
        res *= gamma
        res += i
        l.append(res)
    return  list(reversed(l))



@torch.no_grad()
def play(env, agent):
    state = env.reset()
    rewards= 0
    while True:
        env.render()
        action = agent(torch.FloatTensor([state]))[0].item()
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help ='play an episode after training the network')
    parser.add_argument('--save', action='store_true', help ='save trained network')
    parser.add_argument('--baseline', action='store_true', default = False, help ='save trained network')
    parser.add_argument('--steps', type=int, default = 1, help ='reward steps')
    parser.add_argument('--game', type=str, default = 'bowling', help ='game to play:CartPole, CartPole500 and LunarLander')
    args = parser.parse_args()

    params = games[args.game]
    ENV_ID = params['ENV_ID']
    GAMMA = params['GAMMA']
    LR = params['LR']
    ENTROPY_BETA = params['ENTROPY_BETA']
    STEPS = args.steps # If > 1 -> ptan experiecne calculates the discounted rewards for you. All you need to do is calculate batch baseline 
    if STEPS > 1:
        args.baseline = True
    PLAY_EPISODES = params['PLAY_EPISODES']
    BOUND = params['BOUND']

    print("*" * 15, args.game, "*" * 15)
    env = gym.make(ENV_ID)
    net = Net(env.observation_space.shape, env.action_space.n)
    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(net, selector, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=STEPS)

    'MAKE SURE TO USE ADAM AS IT IS MORE STABLE FOR THIS LEARNING ALGORITHM'
    'I tried using SGD but it took +500 epochs to solve, when ADAM solves it in under 10 seconds and 43 epochs'
    optimizer = torch.optim.Adam(net.parameters(), lr= LR)

    total_rewards = deque(maxlen=100)
    epoch = 0
    loss = 0.0
    train_episode = 0
    start_time = datetime.now()
    start_print = time()
    print(net)
    bs,ba,br = [],[],[]
    q_val = []
    for exp in exp_source:
        bs.append(exp.state)
        ba.append(int(exp.action))
        br.append(exp.reward)
        if exp.last_state is None:
            if STEPS == 1:
                q_val.extend(discount_rewards(br,GAMMA))
            else:
                q_val.extend(br)
            br.clear()
            train_episode +=1
        
        if train_episode < PLAY_EPISODES:
            continue
        epoch += 1
        reward = exp_source.pop_total_rewards()
        total_rewards.extend(reward)
        if not total_rewards: print(len(q_val), train_episode)
        mean = np.mean(total_rewards)
        if time() - start_print > 1:
            print(f'Epoch:{epoch:5}, Loss: {loss:7.2f}, batch rewards:{np.mean(reward):7.2f}, mean rewards:{mean:7.2f}')
            start_print = time()
        if mean > BOUND:
            duration = timedelta(seconds=(datetime.now()-start_time).seconds)
            print(f'Solved in {duration}')
            break
        
        states_v = torch.FloatTensor(np.array(bs, copy=False))
        actions_v = torch.tensor(ba)
        ref_q_val = calc_q_values(q_val, baseline = args.baseline)
        ref_q_val_v = torch.FloatTensor(np.array(ref_q_val, copy=False)) 
        bs.clear();ba.clear();br.clear();q_val.clear()      
        train_episode = 0

        optimizer.zero_grad()
        logit_v = net(states_v)
        log_prob_v = F.log_softmax(logit_v, dim=1)
        prob_v_a = log_prob_v[range(len(actions_v)), actions_v]
        policy_loss = - (ref_q_val_v * prob_v_a).mean(dim=-1)

        #entropy loss
        prob_v = F.softmax(logit_v, dim=1)
        ent = - (prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss = - ENTROPY_BETA * ent

        loss = policy_loss + entropy_loss
        loss.backward()
        optimizer.step()

    if args.save: torch.save(net.state_dict(), 'trainedModels/policyRL.dat')
    if args.play: 
        # from lib.common import play_episode
        play(env,agent)
