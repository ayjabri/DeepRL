#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 06:07:58 2021

@author: ayman

GRU
"""

import os
import gym
import ptan
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from lib import hyperparameters
from tensorboardX import SummaryWriter


class DRQN(nn.Module):
    def __init__(self, shape, actions, hidden_size=40, num_layers=2, device='cpu'):
        super().__init__()
        self.shape = shape
        self.actions = actions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.gru = nn.GRU(shape[0], hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, actions)
        self.act = nn.ReLU()

        self.to(device)

    def forward(self, x, h):
        fx= x.unsqueeze(1)
        self.gru.flatten_parameters()
        out, h = self.gru(fx,h)
        out = self.fc(self.act(out[:,-1,:]))
        return out, h

    def init_hidden(self,batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)


@torch.no_grad()
def play(env, agent):
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        action = agent(torch.FloatTensor([state]))[0].item()
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()



def unpack_dqn_batch(batch):

    states, actions, rewards, dones, last_states = [], [], [], [], []
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


def calc_dqn_loss(batch, net, tgt_net, gamma, device='cuda'):

    states, actions, rewards, dones, last_states = unpack_dqn_batch(batch)
    states_v = torch.FloatTensor(states).to(device)
    last_states_v = torch.FloatTensor(last_states).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
    size = len(actions)
    qval_v = net(states_v)
    qval_a = qval_v[range(size), actions]
    with torch.no_grad():
        next_qval = tgt_net(last_states_v)
        best_next_qval = next_qval.max(dim=1)[0]
        best_next_qval[dones] = 0
    future_rewards = rewards_v + gamma * best_next_qval
    # F.smooth_l1_loss(qval_a,future_rewards) #alternative loss function if grandients exploded!
    return nn.functional.mse_loss(qval_a, future_rewards)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', default=False,help='play and episode once finished training')
    parser.add_argument('--save', '-s', action='store_true', default=True,help='Save a copy of the trained network in current directory as "lunar_dqn.dat"')
    parser.add_argument('--env', default='cartpole1',help='name of the game: lander, cartpole, freeway')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = hyperparameters.HYPERPARAMS[args.env]
    params.LR = 1e-3
    env = gym.make(params.ENV_ID)
    shape = env.observation_space.shape
    actions = env.action_space.n

    # User Deep Recurrent Q-Learning network
    net = DRQN(shape, actions, device =device)

    net_type = 'DRQN'
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.EPS_START, params.EPS_END,
                                              params.EPS_FRAMES)
    agent = ptan.agent.DQNAgent(lambda x: net(x, net.init_hidden(1))[0], selector,
                                device=device, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, params.GAMMA,
                                                           steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, params.BUFFER_SIZE)

    comment = f'_{args.env}_DRQN'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('runs', current_time + '_' + comment)
    writer = SummaryWriter(logdir=logdir)

    # s_v = ptan.agent.float32_preprocessor([env.reset()])
    # writer.add_graph(net, s_v)
    writer.add_text(args.env, net_type + str(params))
    optimizer = torch.optim.Adam(net.parameters(), lr=params.LR)

    st = datetime.now()
    frame = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += 1
            eps_tracker.frame(frame)
            buffer.populate(1)
            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                mean = tracker.reward(
                    new_reward[0], frame, epsilon=selector.epsilon)
                if mean:
                    if mean > params.BOUND_SOLVE:
                        duration = timedelta(
                            seconds=(datetime.now()-st).seconds)
                        print(f'Solved in {duration}')
                        if args.save:
                            f_name = ''.join(env.spec.id,'_',net_type,'.dat')
                            torch.save(net.state_dict(), f_name)
                        if args.play:
                            play(env, agent)
                        break
            if len(buffer) < params.INIT_REPLAY:
                continue
            optimizer.zero_grad()
            batch = buffer.sample(params.BATCH_SIZE)
            loss = calc_dqn_loss(batch, lambda x: net(x, net.init_hidden(params.BATCH_SIZE))[0],
                                       lambda x: tgt_net.target_model(x, net.init_hidden(params.BATCH_SIZE))[0],
                                        params.GAMMA, device=device)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=frame,
                              display_name='Mean Square Errors Loss')
            if frame % params.SYNC_NET == 0:
                tgt_net.sync()
