#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on Haoran Tang -and others- paper:
    # Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning
    
Created on Sat Jun 19 07:30:49 2021

@author: ayman.aljabri@gmail.com
"""

import os
import torch
import torch.nn as nn
import numpy as np
from lib import models
from hashlib import md5 as h_fun

PATH = '/home/ayman/workspace/data/AutoEncoder'

def collect_observations(env, N):
    states, rewards, dones = [],[],[]
    state = env.reset()
    reward = 0
    done=False
    while len(states)<N:
        states.append(np.array(state, copy=False))
        rewards.append(reward)
        dones.append(done)
        action= env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
    return np.array(states,copy=False), np.array(rewards), np.array(dones)
            
        

if __name__ == '__main__':
    # Hyperparameters
    env_name = 'FrostbiteNoFrameskip-v4'
    f_name = 'FrostbiteRandStates.dat'
    batch_size = 512
    alpha = 0.5
    learning_rate = 1e-3
    device = torch.device('cuda:0')
    num_epochs = 20


    train_path = os.path.join(PATH, f_name)
    # Prepare data
    dataset = torch.tensor(torch.load(train_path))/255
    rewardset = torch.tensor(torch.load(os.path.join(PATH, 'FrostbiteRewards.dat')))
    train_set = torch.utils.data.TensorDataset(dataset, rewardset)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)
    
    state,r = next(iter(train_set))
    shape = state.shape
    net = models.AutoEncoder(shape,alpha,batch_size).to(device)
    
    # evaluate(net, test_loader, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fun = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for states,_ in train_loader:
            states = states.to(device)
            optimizer.zero_grad()
            output = net(states)
            loss_v = loss_fun(output, states)
            loss_v.backward()
            optimizer.step()
            total_loss += loss_v.item()
        # accuracy = evaluate(net, test_loader, device)
        # , Testing Accuracy: {accuracy:.2f}%')
        print(f'Epoch:{epoch:3}, Loss:{total_loss:7.3f}')

