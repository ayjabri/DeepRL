# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def calc_reinforce_loss(batch, net, device = 'cpu'):
    states_v = torch.FloatTensor(batch[0]).to(device)
    actions_v = torch.LongTensor(batch[1]).to(device)
    ref_v = torch.FloatTensor(batch[2]).to(device)

    prob_v = net(states_v)
    log_prob_v = F.log_softmax(prob_v, dim=1)[range(
        len(actions_v)), actions_v]  # logsoft taken actions
    loss_v = (- ref_v * log_prob_v).mean()
    return loss_v
