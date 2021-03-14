#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 05:42:38 2020

@author: ayman
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
from lib import model,utils,data
from tensorboardX import SummaryWriter
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
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    args = parser.parse_args()

    params = data.HYPERPARAMS[args.env]
    device = 'cuda' if args.cuda else 'cpu'
    if args.steps:
        params.steps = args.steps

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    act_net = model.ActorNet(params.obs_size, params.act_size, params.high_action).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    crt_net = model.CriticNet(params).to(device)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    print("D4PG Actor Net", act_net)
    print("D4PG Crititc Net", crt_net)

    envs = []
    for _ in range(params.n_envs):
        env = gym.make(params.env_id)
        env.seed(params.seed)
        envs.append(env)

    agent = model.DDPGAgent(act_net,device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.gamma,steps_count=params.steps)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params.buffer_size)

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

    dz= (params.vmax - params.vmin)/(params.natoms - 1)
    total_rewards = []
    pt = time()
    st = datetime.now()
    frame = 0
    last_frame = 0
    episode = 0
    actor_loss, critic_loss = None, None
    while True:
        frame +=1
        buffer.populate(params.n_envs)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            episode += 1
            total_rewards.append(new_reward[0])
            mean_reward = np.mean(total_rewards[-100:])
            if time() - pt > 1:
                speed = (frame - last_frame)/(time()-pt)
                print(
                    f"{frame:,}: done {episode} episodes,mean reward {mean_reward:6.3f}, speed {speed:6.0f} fps", flush=True)
                pt = time()
                last_frame = frame

            if mean_reward > params.bound_solve:
                duration = timedelta(seconds=(datetime.now()-st).seconds)
                print(
                    f'Solved in {duration} with mean reward {mean_reward:6.3f}', flush=True)
                if args.write:
                    writer.add_text("Completed in", str(duration))
                    writer.close()
                break
        if len(buffer) < params.init_replay:
            continue

        batch = buffer.sample(params.batch_size)
        s,a,r,d,l = utils.unpack_dqn_batch(batch)
        states = torch.tensor(s).to(device)
        actions = torch.tensor(a).to(device)
        rewards = torch.tensor(r).to(device)
        dones = torch.BoolTensor(d).to(device)
        last_states = torch.tensor(l).to(device)

        # Train Critic
        crt_optim.zero_grad()
        q_sa_dist = crt_net(states, actions)
        a_last = tgt_act_net.target_model(last_states)
        q_la_dist = tgt_crt_net.target_model(last_states, a_last)
        q_la_prob = torch.softmax(q_la_dist, dim=1).cpu().data.numpy()
        q_proj_prob = utils.calc_proj_dist(q_la_prob, r,d,params.vmin,
                                   params.vmax,params.natoms,dz,params.gamma)
        q_proj_prob_v = torch.FloatTensor(q_proj_prob).to(device)
        critic_loss = - torch.log_softmax(q_sa_dist, dim = 1) * q_proj_prob_v
        critic_loss = critic_loss.sum(dim=1).mean()
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
                              global_step=frame)
            writer.add_scalar('Actor Loss', - actor_loss,
                              global_step=frame)
            writer.add_scalar('Critic Loss', critic_loss,
                              global_step=frame)
            for name, param in act_net.named_parameters():
                writer.add_histogram(name, param, global_step=frame)

    if args.save:
        fname = f'ddpg_{args.env}_{str(st.date())}.dat'
        torch.save(act_net.state_dict(), fname)
    if args.play:
        play(env, agent)
