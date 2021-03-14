#!/usr/bin/env python
# coding: utf-8
'''
Author Ayman Al jabri
Reinforce Method:
Is one of the simplist Policy_gradient methods. It uses the same formula:
    loss= - sum(Q(s,a) log(pi(s,a))) ----- Where Q(s,a): is the gradient scale. 
                                           Q(s,a) = discounted rewards or sum(gamm**i * ri)
steps:
    1.Initialize the network with random weights
    2. Play N full episodes, saving their (𝑠,𝑎,𝑟,𝑠′) transitions
    3. For every step, t, of every episode, k, calculate the discounted total reward for
        subsequent steps: 𝑄(𝑘,𝑡) = Σ 𝛾^𝑖 * 𝑟_𝑖
    4. Calculate the loss function for all transitions: ℒ = −Σ𝑄(𝑘,𝑡) log 𝜋(𝑠,𝑎)
    5. Perform an SGD update of weights, minimizing the loss (Use Adam instead - much faster)
    6. Repeat from step 2 until converged

Usually solve in 440 episodes within 0:00:09
'''
import os
import gym
import ptan
import numpy as np
import argparse


import torch
from datetime import datetime, timedelta
from time import time
from lib import model, utils, hyperparameters
from tensorboardX import SummaryWriter

@torch.no_grad()
def play(env, agent):
    state= env.reset()
    rewards = 0
    while True:
        env.render()
        state_v = torch.FloatTensor([state])
        action = agent(state_v)[0]
        state,r,done,_= env.step(action.item())
        rewards+=r
        if done:
            print(rewards)
            break
    env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--play', action='store_true', help='Play an episode after training is complete')
    parser.add_argument('--save',action='store_true', default=False, help='Store a copy of the network')
    parser.add_argument('--env', default='lander', help='Game name: cartpole, cartpole1, lander, freeway..etc')
    parser.add_argument('--episodes', type=int, help='train N episodes per batch')
    parser.add_argument('--cuda', default=True, action='store_true', help='Use GPU')
    args = parser.parse_args()
    
    params = hyperparameters.HYPERPARAMS[args.env]
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    if args.episodes: params.steps = args.episodes
    env = gym.make(params.env_id)
    net = model.RLNet(params.obs_size, params.act_size).to(device)
    print(net)
    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor,device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, params.gamma)
    generator = model.BatchGenerator(exp_source,params.steps,params)
    
    comment = f'_{args.env}_Reinforce_{params.steps}_episodes'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('runs', current_time + '_' + comment)
    writer = SummaryWriter(logdir=logdir)
    
    'MAKE SURE TO USE ADAM OPTIMIZER; IT IS THE MOST STABLE FOR THIS METHOD'
    'I tried using SGD but it took +500 epochs to solve while ADAM solves it in under 10 seconds and 43 epochs'
    optimizer = torch.optim.Adam(net.parameters(), lr = params.lr)
    
    loss_v = 0.0
    pt = time()
    st = datetime.now()
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for batch in generator:
            for n in range(params.steps):
                reward = generator._total_rewards[n-params.steps]
                frame = generator._end_episode_frames[n-params.steps]
                tracker.reward(reward,frame)
            mean = np.mean(generator._total_rewards[-100:])
            if mean > params.bound_solve:
                print('Solved in {} episodes within {}'.format(generator.episodes, timedelta(seconds=(datetime.now()-st).seconds)))
                break
            optimizer.zero_grad()
            loss_v = utils.calc_reinforce_loss(batch, net, device)
            loss_v.backward()
            writer.add_scalar('loss', loss_v.item(),global_step=generator.frame,display_name=args.env)
            optimizer.step()
        
    
    if args.save:
        fname = f'reinforce_{args.env}_{str(st.date())}.dat'
        torch.save(net.state_dict(), fname)
    if args.play: play(env,agent)
    
