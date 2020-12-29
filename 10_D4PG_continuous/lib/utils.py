#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 05:43:04 2020

@author: ayman
"""
import numpy as np
import torch


def calc_proj_dist(prob, rewards, dones, vmin, vmax, natoms, dz, gamma):
    """
    Redistribute the probability of Q(s,a) based on the value of rewards using Bellman operator
    𝑍(𝑥, 𝑎) = 𝑅(𝑥, 𝑎) + 𝛾𝑍(𝑥′, 𝑎′) 

    Parameters:
    -----------
    prob: probability distribution of Q values over supporting values

    """
    proj = np.zeros(prob.shape)
    for atom in range(natoms):
        v = rewards + (vmin + atom * dz) * gamma
        v = np.maximum(vmin, np.minimum(v, vmax))
        idx = (v-vmin)/dz
        l = np.floor(idx).astype(int)
        u = np.ceil(idx).astype(int)
        eq_mask = l == u
        proj[eq_mask, l[eq_mask]] += prob[eq_mask, atom]
        neq_mask = l != u
        proj[neq_mask, l[neq_mask]] += prob[neq_mask, atom] * \
            (idx - l)[neq_mask]
        proj[neq_mask, l[neq_mask]] += prob[neq_mask, atom] * \
            (u - idx)[neq_mask]
    if dones.any():
        proj[dones] = 0.
        d_v = np.maximum(vmin, np.minimum(rewards[dones], vmax))
        d_idx = (d_v-vmin)/dz
        proj[dones, d_idx.astype(int)] = 1.0
    return proj


def unpack_dqn_batch(batch):
    """
    Definition: unpack_dqn_batch(batch).

    Parameters
    ----------
    batch : a list contains a namedtuples of (state,action,reward,last_state)

    Returns
    -------
    states:float32

    actions:int

    rewards:float64

    dones:bool

    last_states:float32

    """

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
