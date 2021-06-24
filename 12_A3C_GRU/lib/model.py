#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:51:53 2021

@author: Ayman Jabri
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod


class A3CGru(nn.Module):
    def __init__(self, shape, actions, device='cpu', hidden_size=512, num_layers=1, dropout=0.2):
        super().__init__()

        self.shape = shape
        self.actions = actions
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0

        self.conv1 = nn.Conv2d(shape[0], 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.calc_conv_size()
        self.gru = nn.GRU(self.conv_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=self.dropout)

        self.value = nn.Linear(hidden_size, 1)
        self.policy = nn.Linear(hidden_size, actions)

        self.init_biases()
        self.to(device)

    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out = self.fwd_conv(x)
        out = out.view(-1, 1, self.conv_size)
        self.gru.flatten_parameters()
        out, hx = self.gru(out, hx)
        out = out[:, -1, :]
        value = self.value(out)
        policy = self.policy(out)
        return policy, value, hx

    def fwd_conv(self, obs):
        o_ = F.leaky_relu(self.pool1(self.conv1(obs)))
        o_ = F.leaky_relu(self.pool2(self.conv2(o_)))
        o_ = F.leaky_relu(self.pool3(self.conv3(o_)))
        o_ = F.leaky_relu(self.pool4(self.conv4(o_)))
        return o_

    def calc_conv_size(self):
        cs = self.fwd_conv(torch.zeros((1, *self.shape))).shape
        self.conv_size = prod(cs[1:])
        pass

    def init_biases(self):
        for name, p in self.named_parameters():
            if 'bias' in name:
                p.data.zero_()

                

class A3CGruSimple(nn.Module):
    def __init__(self, shape, actions, device='cpu', hidden_size=128, num_layers=2, dropout=0.2, sequence=5):
        super().__init__()

        self.shape = shape
        self.actions = actions
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence = sequence
        self.dropout = dropout if num_layers > 1 else 0

        self.gru = nn.GRU(shape[0], hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=self.dropout)

        self.value = nn.Linear(hidden_size*sequence, 1)
        self.policy = nn.Linear(hidden_size*sequence, actions)

        self.to(device)

    def forward(self, x, hx=None):
        if hx is None:
            # hx -> (Layers, batch, hidden)
            hx = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # self.gru.flatten_parameters()
        out, hx = self.gru(x, hx)
        out = out.view(-1, out.size(1)*out.size(2))
        value = self.value(out)
        policy = self.policy(out)
        return policy, value, hx
