#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 07:06:03 2020

@author: Ayman Al Jabri

Deep Deterministic Policy Gradient (DDPG) method is from the Actor-Critic family
but it is slightly different in the way it uses Critic network.
In A2C the critic is used to get a baseline for our discounted rewards, while in
DDPG it returns Q(s,a) value which represents the total discounted rewards
of action "a" in state "s".
The policy is deterministic, which means the actor returns the action directly from
Actor network. Unlike A2C which is stochastic i.e. returns probability distribution
parameters of the action (mean and variance).

"""
import os
import torch
import gym
import ptan
import argparse
import torch.nn.utils as nn_utils
import numpy as np
from time import time
from datetime import datetime, timedelta
from lib import model, mp_utils, data
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.nn.functional as F

@torch.no_grad()
def play(env, act_net, render=False):
    """Play an episode using trained agent."""
    state = env.reset()
    rewards = 0
    while True:
        if render: env.render()
        state_v = torch.FloatTensor(state)
        actions = act_net(state_v).data.numpy()
        state, r, done, _ = env.step(actions)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true',
                        help='Play an episode after training is complete')
    parser.add_argument('--save', action='store_true',
                        default=False, help='Store a copy of the network')
    parser.add_argument('--env', default='lander',
                        help='Game name: cartpole, cartpole500, lander, freeway..etc')
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

    EPSILON = 0.2
    act_net = model.ActorNet(params.obs_size, params.act_size)
    act_net.share_memory()
    tgt_act_net = ptan.agent.TargetNet(act_net)
    crt_net = model.CriticNet(params.obs_size, params.act_size)
    crt_net.share_memory()
    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    print("DDPG Actor Net", act_net)
    print("DDPG Crititc Net", crt_net)

    exp_queue = mp.Queue(maxsize=params.n_mp)
    procs = []
    for _ in range(params.n_mp):
        proc = mp.Process(target=mp_utils.data_fun,args=(act_net,exp_queue,params))
        proc.start()
        procs.append(proc)

    buffer = ptan.experience.ExperienceReplayBuffer(None, params.buffer_size)
    generator = mp_utils.MPBatchGenerator(buffer, exp_queue,params.init_replay, params.batch_size, 1)

    if args.write:
        train_mode = f'batches_{params.batch_size}' if args.batch else f'episodes_{args.episodes}'
        comment = f'_{args.env}_A2C_{train_mode}'
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join('runs', current_time + '_' + comment)
        writer = SummaryWriter(logdir=logdir)

    # 'MAKE SURE YOU USE ADAM OPTIMIZER AS IT IS THE MOST STABLE FOR THIS LEARNING ALGORITHM'
    # 'I tried using SGD but it took +500 epochs to solve while ADAM solves it in under 10 seconds and 43 epochs'
    act_optim = torch.optim.Adam(act_net.parameters(), lr=params.lr)
    crt_optim = torch.optim.Adam(crt_net.parameters(), lr=1e-4)

    # total_rewards = []
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
                f"{frame:,}: done {episode} episodes,mean reward {mean_reward:6.3f}, speed {speed}fps", flush=True)
            pt = time()

        if mean_reward > params.bound_solve:
            duration = timedelta(seconds=(datetime.now()-st).seconds)
            print(
                f'Solved in {duration} with mean reward {mean_reward:6.3f}', flush=True)
            if args.write:
                writer.add_text("Completed in", str(duration))
                writer.close()
            break

        s,a,r,d,l = mp_utils.unpack_dqn_batch(batch)
        states = torch.FloatTensor(s)
        actions = torch.FloatTensor(a)
        rewards = torch.FloatTensor(r)
        dones = torch.BoolTensor(d)
        last_states = torch.FloatTensor(l)
        # Train Critic
        crt_optim.zero_grad()
        q_sa = crt_net(states, actions)
        a_last = tgt_act_net.target_model(last_states)
        q_sa_last = tgt_crt_net.target_model(last_states, a_last)
        q_sa_last[dones] = 0.0
        q_ref_val = rewards.unsqueeze(-1) + q_sa_last * params.gamma
        critic_loss = F.mse_loss(q_sa, q_ref_val.detach())
        critic_loss.backward()
        crt_optim.step()

        # Train Actor
        act_optim.zero_grad()
        a_curr = act_net(states)
        actor_loss = (- crt_net(states, a_curr)).mean()
        actor_loss.backward()
        act_optim.step()

        tgt_act_net.alpha_sync(alpha=1-1e-3)
        tgt_crt_net.alpha_sync(alpha=1-1e-3)


        # write stuff in Tensorboard
        if args.write:
            writer.add_scalar('Mean Rewards', mean_reward,
                              global_step=generator.frame)
            writer.add_scalar('Actor Loss', - actor_loss,
                              global_step=generator.frame)
            writer.add_scalar('Critic Loss', critic_loss,
                              global_step=generator.frame)
            for name, param in act_net.named_parameters():
                writer.add_histogram(name, param, global_step=generator.frame)

    if args.save:
        fname = f'ddpg_{args.env}_{str(st.date())}.dat'
        torch.save(act_net.state_dict(), fname)
    if args.play:
        env = gym.make(params.env_id)
        agent = mp_utils(act_net, epsilon=0)
        play(env, agent)
    for p in procs:
        p.terminate()
        p.join()
