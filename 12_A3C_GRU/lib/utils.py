#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:07:58 2021

@author: Ayman Jabri

Trian function: take n number of steps, then calculates rewards, loss and backpropogate
Test function:  play full episode and evalute rewards, mean

"""
import copy
import torch
from .wrappers import create_atari
from .agents import A3CGruAgent



def share_grads(local_net, shared_net):
    for local_param, shared_param in zip(local_net.parameters(),
                                   shared_net.parameters()):
        shared_param.data.grad = local_param.data.grad



def train(p_number, params, shared_net, optimizer, frames, episodes, steps=20):
    """Train function used for Multiprocessing"""

    ################### train function ######################
    torch.manual_seed(params.seed + p_number)
    env = create_atari(params)
    env.unwrapped.seed(params.seed + p_number)
    # create a duplicate network
    net = copy.deepcopy(shared_net)
    device = net.device
    agent = A3CGruAgent(net, env, frames, episodes)
    while True:
        net.load_state_dict(shared_net.state_dict())

        for step in range(steps):
            agent.step()
            if agent.done:break

        R = torch.zeros(1,1).to(device)
        if not agent.done:
            _, value , _ = agent.model(agent.state, None)
            R = value.data
        agent.values.append(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(device)
        for i in reversed(range(len(agent.rewards))):
            R = R * params.gamma + agent.rewards[i]
            advantage = R - agent.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = agent.rewards[i] + params.gamma * \
                agent.values[i + 1].data - agent.values[i].data

            gae = gae * params.gamma * params.tau + delta_t

            policy_loss = policy_loss - agent.log_probs[i] * \
                gae - 0.01 * agent.entropies[i]

        agent.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        share_grads(agent.model, shared_net)
        optimizer.step()
        agent.clear()
        if agent.done:
            agent.reset()
        # if episodes.value >= 1:
        #     break


def test(agent, queue=None):
    # while not queue.full():
    #     agent.play()
    #     queue.put(agent.get_total_rewards())
    agent.play()
    return agent.get_total_rewards()

