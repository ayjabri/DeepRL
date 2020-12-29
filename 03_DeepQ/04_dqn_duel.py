# -*- coding: utf-8 -*-
"""
Duel DQN:
--------
You have seen the quantity V(s) before, as it was the core of the value iteration method 
from Tabular Learning and the Bellman Equation. It is just equal to the discounted expected
reward achievable from this state. The advantage A(s, a) is supposed to bridge the gap from
A(s) to Q(s, a), as, by definition:
    Q(s, a) = V(s) + A(s, a)
In other words, the advantage A(s, a) is just the delta, saying how much extra reward some 
particular action from the state brings us. The advantage could be positive or negative and,
in general, can have any magnitude. For example, at some tipping point, the choice of one 
action over another can cost us a lot of the total reward. (Max Lapan)

What's New:
---------
Only change the neural network by adding two heads in addition to the input layer(s), or conv:
    Value: single output
    Advantage: Number of actions
Forward function calculates Q, where:
Q(s,a) = V(s) + A(s, a) - 1/n * sum(A(s,k))

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
                        help='name of the game: lander, cartpole, freeway')
    args = parser.parse_args()

    params = hyperparameters.HYPERPARAMS[args.env]
    env = gym.make(params.ENV_ID)
    net = model.DuelDQNet(
        params.OBS_SIZE, params.ACT_SIZE, params.HID_SIZE)  # 1

    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector()
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.EPS_START, params.EPS_END,
                                              params.EPS_FRAMES)
    agent = ptan.agent.DQNAgent(
        net, selector, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, params.GAMMA)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, params.BUFFER_SIZE)

    comment = f'_{args.env}_Duel_DQN'
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
            frame += 1
            eps_tracker.frame(frame)
            buffer.populate(1)
            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                mean = tracker.reward(
                    new_reward[0], frame, epsilon=selector.epsilon)
                if mean:
                    if mean > params.BOUND_SOLVE:
                        duration = timedelta(
                            seconds=(datetime.now()-st).seconds)
                        print(f'Solved in {duration}')
                        if args.save:
                            torch.save(net.state_dict(),
                                       f'{args.env}_dqn_{args.steps}_steps.dat')
                        if args.play:
                            play(env, agent)
                        break
            if len(buffer) < params.INIT_REPLAY:
                continue
            optimizer.zero_grad()
            batch = buffer.sample(params.BATCH_SIZE)
            loss = utils.calc_dqn_loss(batch, net, tgt_net, params.GAMMA)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=frame,
                              display_name='Mean Square Errors Loss')
            if frame % params.SYNC_NET == 0:
                tgt_net.sync()
