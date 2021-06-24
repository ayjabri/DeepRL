#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:55:47 2021

@author: ayman
"""

from lib import data, utils
from lib.model import A3CGruSimple
from lib.agents import A3CGruAgent

import gym
import sys
import time
import torch

import numpy as np

class Variable():
    pass


SEQUENCE = 5

################### train function ######################
if __name__=="__main__":


    params = data.params["cartpole"]
    params.tau = 1.0
    device = "cuda"
    env = gym.make('CartPole-v1')
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


    for p in procs:
        p.kill()
        p.join()
    sys.exit()
