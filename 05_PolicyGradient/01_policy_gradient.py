#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:57:31 2020
@author: Ayman Al Jabri

Now adays nobody uses plain_vanilla policy gradient but this is for illustration purposes
Policy gradient: is an improvment of 'Reinforce' method. It addresses the problem of exploration. It punishes-
The agent for being certain of the action. it does so by substracting an entropy value from policy loss

Steps:
1- Initialize the network with random weights.
2- Play N full episodes, saving their (s,a,r,s') transitions
3- For every step t of every episode, calculate the total discounted
    rewards of subsequent steps 𝑄(𝑘,𝑡) = Σ 𝛾^𝑖 * 𝑟_𝑖
4- Calculate policy loss = ℒ = −Σ 𝑄(𝑘,𝑡) log 𝜋(𝑠,𝑎)
5- Calculate entropy loss
6- Add both losses together to get total loss
7- Perform Adam update of weights
8- Repeat from step 2
"""
import os
import gym
import ptan
import torch
import argparse
import numpy as np
from time import time

from lib import model, utils, data
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta

ENTROPY_BETA = 0.02


@torch.no_grad()
def play(env, agent):
    """Play an episode using trained agent."""
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        state_v = torch.FloatTensor([state])
        action = agent(state_v)[0]
        state, r, done, _ = env.step(action.item())
        rewards += r
        if done:
            print(rewards)
            break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',
                        help='Play an episode after training is complete')
    parser.add_argument('--save', action='store_true',
                        default=False, help='Store a copy of the network')
    parser.add_argument('--env', default='cartpole500',
                        help='Game name: cartpole, cartpole1, lander, freeway..etc')
    parser.add_argument('--episodes', type=int,
                        help='train N episodes per batch')
    parser.add_argument('--write', action='store_true',
                        default=False, help='write to Tensorboard')
    args = parser.parse_args()

    params = data.HYPERPARAMS[args.env]
    if args.episodes:
        params.steps = args.episodes
    env = gym.make(params.env_id)
    net = model.PGNet(params.obs_size, params.act_size, params.hid_size)
    print(net)
    # selector = model.CategoricalSelector() # experemental
    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(
        net, action_selector=selector, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, params.gamma, steps_count=1)
    generator = model.BatchGenerator(exp_source, params.steps, params)

    if args.write:
        comment = f'_{args.env}_PG_{params.steps}_episodes'
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join('runs', current_time + '_' + comment)
        writer = SummaryWriter(logdir=logdir)

    'MAKE SURE YOU USE ADAM OPTIMIZER AS IT IS THE MOST STABLE FOR THIS LEARNING ALGORITHM'
    'I tried using SGD but it took +500 epochs to solve while ADAM solves it in under 10 seconds and 43 epochs'
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    total_rewards = []
    pt = time()
    st = datetime.now()
    frame = 0
    for batch in generator:
        episode = generator.episode
        mean_reward = np.mean(generator._total_rewards[-100:])
        if time() - pt > 1:
            speed = (generator.frame - frame)/1
            frame = generator.frame
            print(
                f"{frame:,}: done {episode} episodes,mean reward {mean_reward:6.3f}", flush=True)
            pt = time()

        if mean_reward > params.bound_solve:
            duration = timedelta(seconds=(datetime.now()-st).seconds)
            print(f'Solved in {duration}')
            if args.write:
                writer.add_text("Completed in", str(duration))
                writer.close()
            break

        optimizer.zero_grad()
        policy_loss, entropy_loss = utils.calc_pg_losses(
            batch, net, ENTROPY_BETA)
        loss = policy_loss + entropy_loss
        loss.backward()
        optimizer.step()

        # write stuff in Tensorboard
        if args.write:
            writer.add_scalar('Mean Rewards', mean_reward,
                              global_step=generator.frame)
            writer.add_scalar('Entropy Loss', - entropy_loss,
                              global_step=generator.frame)
            writer.add_scalar('Policy Loss', - policy_loss,
                              global_step=generator.frame)
            writer.add_scalar('Loss', - loss, global_step=generator.frame)
            for name, param in net.named_parameters():
                writer.add_histogram(name, param, global_step=generator.frame)

    if args.save:
        fname = f'policygradient_{args.env}_{str(st.date())}.dat'
        torch.save(net.state_dict(), fname)
    if args.play:
        play(env, agent)
