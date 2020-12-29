#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 05:41:23 2020

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
from lib import model, utils, data
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
@torch.no_grad()
def play(env, agent):
    """Play an episode using trained agent."""
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        time.sleep(0.03)
        state = np.array(state)
        state_v = torch.FloatTensor([state])
        action = agent(state_v)[0]
        state, r, done, _ = env.step(action.item())
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
    parser.add_argument('--env', default='riverraid_conv',
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

    ENTROPY_BETA = 0.02
    net = model.A2C_Conv(params.obs_size, params.act_size)
    net.share_memory()
    print("Actor Crititc Net", net)

    exp_queue = mp.Queue(maxsize=params.n_mp)
    procs = []
    for _ in range(params.n_mp):
        proc = mp.Process(target=utils.batch_data_fun,args=(net,exp_queue,params))
        proc.start()
        procs.append(proc)

    generator = model.MPEpisodeGenerator(exp_queue)

    if args.write:
        train_mode = f'batches_{params.batch_size}' if args.batch else f'episodes_{args.episodes}'
        comment = f'_{args.env}_A2C_{train_mode}'
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join('runs', current_time + '_' + comment)
        writer = SummaryWriter(logdir=logdir)

    # 'MAKE SURE YOU USE ADAM OPTIMIZER AS IT IS THE MOST STABLE FOR THIS LEARNING ALGORITHM'
    # 'I tried using SGD but it took +500 epochs to solve while ADAM solves it in under 10 seconds and 43 epochs'
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

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

        optimizer.zero_grad()
        value_loss, policy_loss, entropy_loss = utils.calc_a2c_losses(
            batch, net, params)
        loss = policy_loss + value_loss + entropy_loss
        loss.backward()
        if args.clip:
            nn_utils.clip_grad_norm_(net.parameters(), max_norm=args.clip)
        optimizer.step()

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
        env = gym.make(params.env_id)
        agent = ptan.agent.ActorCriticAgent(
            net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
        play(env, agent)
    for p in procs:
        p.terminate()
        p.join()