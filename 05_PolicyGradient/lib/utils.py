#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:58:22 2020

@author: ayman
"""
import torch
import torch.nn.functional as F


def calc_pg_losses(batch, net, entropy_beta=0.02):
    """
    Calculate Policy and Entropy losses from batch.

    Parameters
    ----------
    batch : BatchGenerator output
        The batch should contain States, Actions and Batch Scale.
    net : nn.Module
        Policy Gradient network.
    entropy_beta : int
        scalar to adjust entorpy value by when calculating entorpy loss.

    Returns
    -------
    policy_loss : Tensor
        DESCRIPTION.
    entropy_loss : Tensor
        DESCRIPTION.
    """
    states_v = torch.FloatTensor(batch[0])
    actions = batch[1]
    batch_scale_v = torch.FloatTensor(batch[2])

    # policy loss
    logits_v = net(states_v)
    log_prob_v = F.log_softmax(logits_v, dim=1)
    # Gather probabilities with taken actions
    log_prob_action_v = batch_scale_v * \
        log_prob_v[range(len(actions)), actions]
    policy_loss = - log_prob_action_v.mean()

    # entropy loss
    probs_v = F.softmax(logits_v, dim=1)
    entropy = - (probs_v * log_prob_v).sum(dim=1).mean()
    entropy_loss = - entropy_beta * entropy
    return policy_loss, entropy_loss
