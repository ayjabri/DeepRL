"""
        *********  Using Multipocessing ***********
Solving Lunar Lander using Deep-Q Learning Method
Network: C51 Network with one hidden layer
Loss : C51 loss function, which is entropy loss between two distributions

Results
-------
I was able to solve this in 9 minutes (dead), which is 2 minutes faster than the exact
same configurations without the multiprocess.

Another run using N_MP as multiplier (i.e. increase batch size):
    Solved in 0:07:45
    Batch_size = 128 x 4 # BATCH_MULT = 4
    LearningRate= 1e-3
    N Environments = 1
    Hidden Size = 348

Notes:
------
This was tested on MacBook-Pro
"""
import torch
import torch.multiprocessing as mp

import os
import gym
import ptan
import argparse
import numpy as np
from time import time
from datetime import datetime, timedelta
from lib import mp_utils, model, hyperparameters, utils
from collections import deque


@torch.no_grad()
def play(env, agent):
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        action = agent(torch.FloatTensor([state]))[0].item()
        state, r, done, _ = env.step(action)
        rewards += r
        if done:
            print(rewards)
            break
    env.close()


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    os.environ['OMP_NUM_THREADS'] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', default=False,
                        help='play and episode once finished training')
    parser.add_argument('--save', '-s', action='store_true', default=True,
                        help='Save a copy of the trained network in current directory as "lunar_dqn.dat"')
    parser.add_argument('--env', default='freeway',
                        help='name of the game: lander, cartpole')
    args = parser.parse_args()

    params = hyperparameters.HYPERPARAMS[args.env]
    support = np.linspace(params.VMIN, params.VMAX, params.NATOMS)
    dz = (params.VMAX-params.VMIN)/(params.NATOMS-1)
    net = model.C51Net(params.OBS_SIZE, params.ACT_SIZE,
                       params.VMIN, params.VMAX, params.NATOMS)
    net.share_memory()
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(
        x), selector, preprocessor=ptan.agent.float32_preprocessor)
    buffer = ptan.experience.ExperienceReplayBuffer(None, params.BUFFER_SIZE)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.LR)
    exp_queue = mp.Queue(params.N_MP * 4)
    proc_list = []
    for n in range(params.N_MP):
        proc = mp.Process(target=mp_utils.c51_data_fun,
                          name=str(n), args=(net, exp_queue, params))
        proc.start()
        proc_list.append(proc)
    generator = mp_utils.MPBatchGenerator(buffer, exp_queue, params.INIT_REPLAY,
                                          params.BATCH_SIZE, params.N_MP)
    pt = time()
    loss = 0.0
    start_time = datetime.now()
    total_rewards = deque(maxlen=100)
    epoch = 0
    for batch in generator:
        new_reward = generator.pop_rewards_idx_eps()
        if new_reward:
            total_rewards.extend(new_reward)
            mean = np.mean(total_rewards)
            if mean > params.BOUND_SOLVE:
                duration = timedelta(
                    seconds=(datetime.now()-start_time).seconds)
                print(f'Solved in {duration}')
                if args.save:
                    torch.save(net.state_dict(), 'lunar_dqn_mp.dat')
                if args.play:
                    env = gym.make(params.ENV_ID)
                    play(env, agent)
                break
            if time()-pt > 1:
                print(
                    f'epoch:{epoch:6,} mean:{mean:7.2f}, loss:{loss:7.3f}, reward: {new_reward[0]:7.2f} epsilon:{generator.epsilon:4.2f}')
                pt = time()

        optimizer.zero_grad()
        loss = utils.calc_dist_loss(batch, net, tgt_net, params.GAMMA, params.VMIN, params.VMAX,
                                    params.NATOMS, dz)
        loss.backward()
        optimizer.step()
        epoch += 1
        if generator.frame % params.SYNC_NET == 0:
            tgt_net.sync()
        del batch
    for p in proc_list:
        p.terminate()
        p.join()
    exp_queue.close()
    exp_queue.join_thread()
