#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 07:32:49 2021

@author: ayman
"""

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Discretizes the state space by forcing it into hash function
    """
    def __init__(self, shape, alpha, device):
        super().__init__()
        assert alpha > 0.25 # per the paper 
        self.shape = shape
        self.alpha = alpha
        self.device = device
        
        self.encoder = nn.Sequential(nn.Conv2d(shape[0], 96, 6, 2),
                                      nn.BatchNorm2d(96),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(96, 96, 6,2,padding=1),
                                      nn.BatchNorm2d(96),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(96, 96, 6, 2, padding=2),
                                      nn.BatchNorm2d(96),
                                      nn.LeakyReLU(),
                                      )
        
        
        self.binary = nn.Sequential(nn.Flatten(start_dim=1),
                                    nn.Linear(7776, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 512),
                                    nn.Sigmoid())
        
        ## decoder
        self.fc_d = nn.Sequential(nn.Linear(512, 2400),
                                  nn.LeakyReLU())
        
        self.decoder = nn.Sequential(nn.ConvTranspose2d(96, 96, 7, 3),
                                     nn.BatchNorm2d(96),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(96, 96, 6, 2),
                                     nn.BatchNorm2d(96),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(96, 1, 6, 2, padding=2),
                                     nn.BatchNorm2d(1),
                                     nn.Sigmoid(),
                                     )
        
        # z = torch.zeros(self.batch_size, 512)
        # self.register_buffer('injected_noise', z)
    
    def forward(self, x):
        y = self.encode(x)
        y = self.decode(y)
        return y
    
    def encode(self,x):
        x = x.float()/255
        c = self.encoder(x)
        return torch.round(self.binary(c))
    
    def decode(self, e):
        z = torch.zeros(e.size(0), 512).to(self.device)
        e = e + z
        e = self.fc_d(e)
        return self.decoder(e.view(-1,96, 5, 5))
    