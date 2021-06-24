#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:01:49 2021

@author: ayman
"""

from lib import data, utils
from lib.wrappers import create_atari
from lib.model import A3CGru
from lib.agents import A3CGruAgent

import os
import sys
import time
import torch
import torch.multiprocessing as mp
import numpy as np

class Variable():
    pass


################### train function ######################
if __name__=="__main__":
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = "1"

    FORKS = 6

    params = data.params["pong"]
    params.tau = 1.0
    device = "cuda"
    env = create_atari(params)
    shape = env.observation_space.shape
    actions = env.action_space.n
    shared_net = A3CGru(shape, actions, device="cuda")
    shared_net.share_memory()

    frames_test, episodes_test = Variable(), Variable()
    test_agent = A3CGruAgent(shared_net, env, frames_test, episodes_test)

    optimizer = torch.optim.Adam(shared_net.parameters(), lr=params.lr)

    frames = mp.Value("i", 0)
    episodes = mp.Value("i", 0)
    rewards_queue = mp.Queue(maxsize=2)

    steps = 20
    procs = []
    for i in range(FORKS):
        p = mp.Process(target=utils.train, args=(i, params, shared_net, optimizer, frames, episodes, steps))
        p.start()
        procs.append(p)

    # p_test = mp.Process(target=utils.test, args=(test_agent, rewards_queue))
    # p_test.start()

    start = time.process_time()
    total_rewards = []
    fps = 0
    print(shared_net)

    while True:
        if  (time.process_time() - start) > 15: #and not rewards_queue.empty():
            delta = (time.process_time() - start)
            rewards = utils.test(test_agent) #rewards_queue.get()
            total_rewards.append(rewards)
            mean = np.mean(total_rewards[-100:])
            speed = (frames.value - fps)/delta
            print(f"done {frames.value:7}, episodes:{episodes.value:5}, test rewards:{rewards}, mean:{mean:5.3f}, speed {speed:.2f} fps")
            start = time.process_time()
            fps = frames.value
            if mean > params.solve_rewards:
                print('Solved!')
                break

    # p_test.kill()
    # p_test.join()
    for p in procs:
        p.kill()
        p.join()
    sys.exit()

# torch.manual_seed(params.seed + p_number)
# env = create_atari(params)
# env.unwrapped.seed(params.seed + p_number)
# # create a duplicate network
# net = copy.deepcopy(shared_net)
# device = net.device
# agent = A3CGruAgent(net, env, frames, episodes)
# while True:
#     net.load_state_dict(shared_net.state_dict())

#     for step in range(steps):
#         agent.step()
#         if agent.done:break

#     R = torch.zeros(1,1).to(device)
#     if not agent.done:
#         _, value , _ = agent.model(agent.state, agent.hx)
#         R = value.data
#     agent.values.append(R)

#     policy_loss = 0
#     value_loss = 0
#     gae = torch.zeros(1, 1).to(device)
#     for i in reversed(range(len(agent.rewards))):
#         R = R * params.gamma + agent.rewards[i]
#         advantage = R - agent.values[i]
#         value_loss = value_loss + 0.5 * advantage.pow(2)

#         # Generalized Advantage Estimataion
#         delta_t = agent.rewards[i] + params.gamma * \
#             agent.values[i + 1].data - agent.values[i].data

#         gae = gae * params.gamma * params.tau + delta_t

#         policy_loss = policy_loss - agent.log_probs[i] * \
#             gae - 0.01 * agent.entropies[i]

#     agent.model.zero_grad()
#     (policy_loss + 0.5 * value_loss).backward()
#     share_grads(agent.model, shared_net)
#     optimizer.step()
#     agent.clear()
#     if agent.done:
#         agent.reset()
#     if episodes.value >= 1:
#         break