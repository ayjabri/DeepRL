#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:42:36 2020

@author: ayman
"""
import gym
import ptan
import torch
import torch.nn.functional as F
from collections import namedtuple
import numpy as np


def unpack_a2c_batch(batch, crt_net, params, two_nets=False, device='cpu'):
    """
    Definition: returns states and actions plut advantage value: Adv(s,a) = Q(s,a) - V(s`).

    Parameters
    ----------
    batch : TYPE
        DESCRIPTION.
    act_net : TYPE
        DESCRIPTION.
    crt_net : TYPE
        DESCRIPTION.
    gamma : TYPE, optional
        DESCRIPTION. The default is 0.99.
    steps : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    states : Tensor
        DESCRIPTION.
    actions : Tensor
        DESCRIPTION.
    Q(s,a) : Tensor
        Q(s,a) = r + gamma ^ steps * V(s`)
        V(s`) = critic_net(s`) --- Only for not_dones last states

    """
    states, actions, rewards, dones, last_states = batch
    not_dones = dones == False  # Switch the dones to no_dones array
    rewards_np = np.copy(rewards)
    if not_dones.any():
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_values_v = crt_net(last_states_v[not_dones])
        if two_nets:
            last_values_np = last_values_v.cpu().data.numpy()
        else:
            last_values_np = last_values_v[1].cpu().data.numpy()
        rewards_np[not_dones] += last_values_np[:, 0] * \
            params.gamma**params.steps
    q_sa_v = torch.FloatTensor(rewards_np)
    states_v = torch.FloatTensor(states)
    actions_v = torch.LongTensor(actions)
    del states, actions, rewards, dones, last_states, rewards_np
    return states_v.to(device), actions_v.to(device), q_sa_v.to(device)


def calc_a2c_losses(batch, act_net, params, crt_net=None, entropy_beta=0.02, device='cpu'):
    """
    Calculate Policy and Entropy losses from batch.

    Parameters
    ----------
    batch : BatchGenerator output
        A tuple contains: (States, Actions, Q(s,a))
    act_net : nn.Module
        Policy Gradient network.
    params : TYPE
        DESCRIPTION.
    crt_net : nn.Module, optional
        DESCRIPTION. The default is None.
    entropy_beta : Int, optional
        scalar to adjust entorpy value when calculating entorpy loss. The default is 0.02.

    Returns
    -------
    policy_loss : Tensor
    value_loss : Tensor
    entropy_loss : Tensor

    """
    # network outputs
    if crt_net is None:
        states_v, actions_v, q_sa_v = unpack_a2c_batch(
            batch, act_net, params, two_nets=False, device=device)
        logits_v, values_v = act_net(states_v)
    else:
        states_v, actions_v, q_sa_v = unpack_a2c_batch(
            batch, crt_net, params, two_nets=True,device=device)
        logits_v = act_net(states_v)
        values_v = crt_net(states_v)
    # Value loss
    value_loss = F.mse_loss(values_v.squeeze(-1), q_sa_v)

    # Policy loss
    log_prob_v = F.log_softmax(logits_v, dim=1)
    # Gather probabilities with taken actions
    log_prob_action_v = log_prob_v[range(len(actions_v)), actions_v]
    adv_v = q_sa_v - values_v.squeeze(-1).detach()
    policy_loss_v = - log_prob_action_v * adv_v
    policy_loss = policy_loss_v.mean()

    # Entropy loss
    probs_v = F.softmax(logits_v, dim=1)
    entropy = - (probs_v * log_prob_v).sum(dim=1).mean()
    entropy_loss = - entropy_beta * entropy
    return value_loss, policy_loss, entropy_loss



EpisodeEnd = namedtuple('EpisodeEnd', 'reward')


def make_env(env_id,stack_frames=4,episodic_life=True,reward_clipping=True):
    env = gym.make(env_id)
    env = ptan.common.wrappers.wrap_dqn(env,stack_frames=4,
                            episodic_life=True, reward_clipping=True)
    return env


def episode_data_fun(net, exp_queue, params, n_episodes=4, device='cpu'):
    envs = [gym.make(params.env_id) for _ in range(params.n_envs)]
    agent = ptan.agent.ActorCriticAgent(net, apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, params.gamma, steps_count=params.steps)
    states, actions, rewards, dones, last_states = [], [], [], [], []
    batch_episodes = 0
    frame = 0
    for exp in exp_source:
        frame += 1
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            exp_queue.put(EpisodeEnd(new_reward[0]))
            batch_episodes += 1
        if batch_episodes >= n_episodes:
            batch = (np.array(states, copy=False),
                     np.array(actions),
                     np.array(rewards),
                     np.array(dones),
                     np.array(last_states, copy=False))
            exp_queue.put(batch)
            states.clear()
            actions.clear()
            rewards.clear()
            last_states.clear()
            dones.clear()
            batch_episodes = 0
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        last_states.append(np.array(exp.last_state, copy=False) if
                           exp.last_state is not None else np.array(exp.state, copy=False))


def batch_data_fun(net, exp_queue, params, device='cpu'):
    envs = [make_env(params.env_id) for _ in range(params.n_envs)]
    agent = ptan.agent.ActorCriticAgent(net, apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, params.gamma, steps_count=params.steps)
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in exp_source:
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            exp_queue.put(EpisodeEnd(new_reward[0]))

        if len(actions) == params.batch_size:
            batch = (np.array(states, copy=False),
                     np.array(actions),
                     np.array(rewards),
                     np.array(dones),
                     np.array(last_states, copy=False))
            exp_queue.put(batch)
            # clear lists
            states.clear()
            actions.clear()
            rewards.clear()
            last_states.clear()
            dones.clear()

        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        last_states.append(np.array(exp.last_state, copy=False) if
                           exp.last_state is not None else np.array(exp.state, copy=False))


def batch_data_fun_conv(net, exp_queue, params):
    envs = [make_env(params.env_id) for _ in range(params.n_envs)]
    agent = ptan.agent.ActorCriticAgent(net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, params.gamma, steps_count=params.steps)
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in exp_source:
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            exp_queue.put(EpisodeEnd(new_reward[0]))

        if len(actions) == params.batch_size:
            batch = (np.array(states, copy=False),
                     np.array(actions),
                     np.array(rewards),
                     np.array(dones),
                     np.array(last_states, copy=False))
            exp_queue.put(batch)
            # clear lists
            states.clear()
            actions.clear()
            rewards.clear()
            last_states.clear()
            dones.clear()

        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        last_states.append(np.array(exp.last_state, copy=False) if
                           exp.last_state is not None else np.array(exp.state, copy=False))