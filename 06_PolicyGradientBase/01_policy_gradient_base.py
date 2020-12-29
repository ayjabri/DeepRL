# -*- coding: utf-8 -*-
'''
Now adays nobody uses plain_vanilla policy gradient but this is for illustration purposes.
Policy gradient: is an improvment of 'Reinforce' method. It addresses two important problems:
a. high gradient variance: it normalizes Q(s,a) value by substracting a baseline from it (the baseline itself
    can be calculated using different methods; average_rewards, moving average or state value. Here we are using average rewards
b. exploration: it improves exploration by punishing the agent when it is too certain of the action. it does so
    by simply substracting entropy value from policy loss

Steps:
1- Initialize the network with random weights.
2- Play N full episodes, saving their (s,a,r,s') transitions
3- For every step t of every episode, calculate the total discounted
    rewards of subsequent steps 𝑄(𝑘,𝑡) = Σ 𝛾^𝑖 * 𝑟_𝑖 - baseline
4- Calculate policy loss = ℒ = −Σ 𝑄(𝑘,𝑡) log 𝜋(𝑠,𝑎)
5- Perform SGD update of weights
6- Repeat from step 2

Usually solves in 0:00:10 after playing 67 epochs when using the same paramteres
in this file it should solve in about 8 seconds. just make sure to use ADAM optimizer
'''
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
    parser.add_argument('--write', action='store_true',
                        default=False, help='write to Tensorboard')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Substract baseline from disc rewards to reduce variance')
    args = parser.parse_args()

    params = data.HYPERPARAMS[args.env]
    ENTROPY_BETA = 0.02
    STEPS = 10  # must set steps to > 1 in order for this to work

    env = gym.make(params.env_id)
    net = model.PGNet(params.obs_size, params.act_size, params.hid_size)
    print(net)
    # selector = model.CategoricalSelector() # experemental
    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(net, action_selector=selector,
                                   apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent,
                                                           params.gamma, steps_count=STEPS)
    generator = model.BatchGenerator(
        exp_source, params, params.batch_size, baseline=args.baseline)

    if args.write:
        base = 'baseline' if args.baseline else 'standard'
        comment = f'_{args.env}_PG_{base}'
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
