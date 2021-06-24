#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 07:36:42 2021

@author: ayman
"""
from collections import defaultdict

def update_rewards(rewards, beta, nhs):
    """
    Add exploration bonus to rewards while training.
    
    Params:
        rewrad: extrinsic reward from the environment (ndarray)
        beta: bonus coefficient >=0 (float)
        nhs: cound of hashed state visited. > 0 (ndarray)
    Note: this is only used for training. Testing is carried out using extrinsic
    rewards.
    """
    return rewards + beta/nhs**0.5


def create_hashtable():
    return defaultdict(lambda:0)
    