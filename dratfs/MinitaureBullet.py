#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:23:20 2020
@author: aymanjabri

Simple introduction to test Pybullet environment
Install PyBullet if you don't have it already:
$pip instal pybullet
"""

import gym
# import pybullet as p
import pybullet_envs

# p.connect(p.GUI)
ENV_ID = 'MinitaurBulletEnv-v0'
RENDER = True

if __name__=="__main__":
    spec = gym.envs.registry.spec(ENV_ID)
    spec._kwargs['render'] = RENDER
    env = gym.make(ENV_ID)
    env.reset()
    rewards = 0
    while True:

        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            print(f'Finished one round with {rewards:.2f} rewards\n Which is the travelled distance minus the effort')
            break
    env.close()
