#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:03:29 2020
@author: Ayman Al Jabri

Steps:
1- Two networks: Critic -> val: 1 output to estimate V(s)) ...
             and Actor -> policy: n_actions output that esitmates the policy pi(a,s)
2- Initialize network parameteres with random values
3- Setup Experience Source to play N steps in the environment using actor network to return actions,....
     and saving states, actions, rewards, dones, last_states

*** The rewrds will be automatically dicounted for N steps ***
4- for i = t-1...t (steps are in reversed order):
    a. calculate discounted rewards in reversed order: ri+gamma*R -> R (ptan library calculates ...
                                                                        this for us for N steps)
    b. Represent total rewards as Q(s,a) = V(s) + Adv(s,a) -> Adv(s,a) = Q(s,a) - V(s) ....
        First: calc Q(s,a) = Sum_{0 to N-1} GAMMA^i * r_i + GAMMA^N * V(s_N) ---- V(s_N)
        is the value head of our network when fed last_state
        second: calc Adv(s,a) = Q(s,a) - V(s) -> use to scale policy loss
    c. accumulate policy gradients: - Σ Adv(s,a) * log(pi(a,s))* (R - V(si)) -> policy_grad
    d. accumulate value gradients: value_grad + MeanSquareError(R, V(si))
5- update the network parameters using the accumulated gradients, moving in the direction of ...
    policy gradient and opposite to value gradient (i.e. subtract policy and add value!)
6- repeat from step 2 until convergence

Problems:
---------
The most obvious problem is the correlation of samples which breaks the i.i.d assumption. To reduce this we
sample from multiple environment using the same policy. But this won't solve it completely!
In the next code we will add another A to A2C to reduce this inefficiency even further.
The method is slower than DQN but it sovles at the end!
"""

import os
import gym
import ptan
import torch
import argparse
import torch.nn.utils as nn_utils
import numpy as np
from time import time
from datetime import datetime, timedelta
from lib import model, utils, data
from tensorboardX import SummaryWriter


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
    parser.add_argument('--env', default='cartpole',
                        help='Game name: cartpole, cartpole1, lander, freeway..etc')
    parser.add_argument('--episodes', type=int, default=4,
                        help='train N episodes per batch')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Train using fixed batch sizes from params')
    parser.add_argument('--steps', type=int, help='Gamma steps')
    parser.add_argument('--write', action='store_true',
                        default=False, help='write to Tensorboard')
    parser.add_argument('--clip', type=float, help='clip grads')
    args = parser.parse_args()

    params = data.HYPERPARAMS[args.env]
    if args.steps:
        params.steps = args.steps

    ENTROPY_BETA = 0.02

    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env_id)
        # env.seed(params.seed)
        envs.append(env)

    act_net = model.ActorNet(params.obs_size, params.act_size, params.hid_size)
    crt_net = model.CriticNet(params.obs_size, params.hid_size)
    print("Actor Net", act_net)
    print("Crititc Net", crt_net)

    # selector = model.CategoricalSelector() # experemental
    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(act_net, action_selector=selector,
                                   apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent,
                                                           params.gamma, steps_count=params.steps)

    generator = model.BatchGenerator(exp_source, params, params.batch_size) if args.batch\
        else model.EpisodeGenerator(exp_source, params, args.episodes)

    if args.write:
        train_mode = f'batches_{params.batch_size}' if args.batch else f'episodes_{args.episodes}'
        comment = f'_{args.env}_A2C_{train_mode}'
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join('runs', current_time + '_' + comment)
        writer = SummaryWriter(logdir=logdir)

    'MAKE SURE YOU USE ADAM OPTIMIZER AS IT IS THE MOST STABLE FOR THIS LEARNING ALGORITHM'
    'I tried using SGD but it took +500 epochs to solve while ADAM solves it in under 10 seconds and 43 epochs'
    act_optimizer = torch.optim.Adam(act_net.parameters(), lr=params.lr)
    crt_optimizer = torch.optim.Adam(crt_net.parameters(), lr=params.lr)

    total_rewards = []
    pt = time()
    st = datetime.now()
    frame = 0
    policy_loss, entropy_loss, loss = None, None, None
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
            print(
                f'Solved in {duration} with mean reward {mean_reward:6.3f}', flush=True)
            if args.write:
                writer.add_text("Completed in", str(duration))
                writer.close()
            break

        act_optimizer.zero_grad()
        crt_optimizer.zero_grad()
        value_loss, policy_loss, entropy_loss = utils.calc_a2c_losses(
            batch, act_net, params, crt_net)
        loss = policy_loss + value_loss + entropy_loss
        loss.backward()
        if args.clip:
            nn_utils.clip_grad_norm_(act_net.parameters(), max_norm=args.clip)
            nn_utils.clip_grad_norm_(crt_net.parameters(), max_norm=args.clip)
        act_optimizer.step()
        act_optimizer.step()

        # write stuff in Tensorboard
        if args.write:
            writer.add_scalar('Mean Rewards', mean_reward,
                              global_step=generator.frame)
            writer.add_scalar('Entropy Loss', - entropy_loss,
                              global_step=generator.frame)
            writer.add_scalar('Value Loss', value_loss,
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
