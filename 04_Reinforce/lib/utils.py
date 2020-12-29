# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def calc_reinforce_loss(batch, net):
    states_v = torch.FloatTensor(batch[0])
    actions_v = torch.LongTensor(batch[1])
    ref_v = torch.FloatTensor(batch[2])

    prob_v = net(states_v)
    log_prob_v = F.log_softmax(prob_v, dim=1)[range(
        len(actions_v)), actions_v]  # logsoft taken actions
    loss_v = (- ref_v * log_prob_v).mean()
    return loss_v
