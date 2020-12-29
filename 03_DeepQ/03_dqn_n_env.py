# -*- coding: utf-8 -*-
"""
N-Environments:
--------
To use N number of environments you must do the following changes to your training loop:
    1- Use envs as a source of observations in ptan experience source instead of env
    2- Increase frame by N instead of 1
    3- Populate N steps in the Buffer
    4- Divide frame by N in Epsilon tracker.frame().
    5- Use common SEED in all environments: Optional 
"""
import os
import gym
import ptan
import argparse
import torch
from datetime import datetime, timedelta
from lib import model, hyperparameters, utils
from tensorboardX import SummaryWriter


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', default=True,
                        help='play and episode once finished training')
    parser.add_argument('--save', '-s', action='store_true', default=False,
                        help='Save a copy of the trained network in current directory as "lunar_dqn.dat"')
    parser.add_argument('--env', default='cartpole1',
                        help='name of the game: lander, cartpole, freeway')  # 1
    parser.add_argument('--n_envs', type=int, help='Number of environments')
    args = parser.parse_args()

    params = hyperparameters.HYPERPARAMS[args.env]

    if args.n_envs:  # 2
        params.N_ENVS = args.n_envs
    envs = []  # 3
    for _ in range(params.N_ENVS):
        env = gym.make(params.ENV_ID)
        envs.append(env)

    net = model.DQNet(params.OBS_SIZE, params.ACT_SIZE, params.HID_SIZE)
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.EPS_START, params.EPS_END,
                                              params.EPS_FRAMES)
    agent = ptan.agent.DQNAgent(
        net, selector, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, params.GAMMA,  # 4
                                                           steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, params.BUFFER_SIZE)

    comment = f'_{args.env}_Basic_DQN_{params.N_ENVS}_envs'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('runs', current_time + '_' + comment)
    writer = SummaryWriter(logdir=logdir)

    s_v = ptan.agent.float32_preprocessor([env.reset()])
    writer.add_graph(net, s_v)
    writer.add_text(args.env, str(params))
    optimizer = torch.optim.SGD(net.parameters(), lr=params.LR)

    st = datetime.now()
    frame = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame += params.N_ENVS  # 5
            eps_tracker.frame(frame/params.N_ENVS)
            buffer.populate(params.N_ENVS)  # 6
            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                mean = tracker.reward(
                    new_reward[0], frame, epsilon=selector.epsilon)  # 7
                if mean:
                    if mean > params.BOUND_SOLVE:
                        duration = timedelta(
                            seconds=(datetime.now()-st).seconds)
                        print(f'Solved in {duration}')
                        if args.save:
                            torch.save(net.state_dict(),
                                       f'{args.env}_dqn_{args.n_envs}_envs.dat')
                        if args.play:
                            play(env, agent)
                        break
            if len(buffer) < params.INIT_REPLAY:
                continue
            optimizer.zero_grad()
            batch = buffer.sample(params.BATCH_SIZE)  # * params.N_ENVS) # 8
            loss = utils.calc_dqn_loss(batch, net, tgt_net, params.GAMMA)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=frame,
                              display_name='Mean Square Errors Loss')
            if frame % params.SYNC_NET == 0:
                tgt_net.sync()
