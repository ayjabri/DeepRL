# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def unpack_dqn_batch(batch):
    """
    Definition: unpack_dqn_batch(batch)

    Unpack a batch of observations

    Parameters
    ----------
    batch : a list contains a namedtuples of (state,action,reward,last_state)

    Returns
    -------       
    states:float32

    actions:int

    rewards:float64

    dones:bool

    last_states:float32

    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    return (np.array(states, copy=False, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(last_states, copy=False, dtype=np.float32))


def calc_dqn_loss(batch, net, tgt_net, gamma, device='cpu'):
    """
    Definition: calc_dqn_loss(batch, net, tgt_net, gamma, device='cpu')

    Parameters
    ----------
    batch : TYPE
        DESCRIPTION.
    net : TYPE
        DESCRIPTION.
    tgt_net : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    states, actions, rewards, dones, last_states = unpack_dqn_batch(batch)
    states_v = torch.FloatTensor(states).to(device)
    last_states_v = torch.FloatTensor(last_states).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    size = len(actions)
    qval_v = net(states_v)
    qval_a = qval_v[range(size), actions]
    with torch.no_grad():
        next_qval = tgt_net.target_model(last_states_v)
        best_next_qval = next_qval.max(dim=1)[0]
        best_next_qval[dones] = 0
    future_rewards = rewards_v + gamma * best_next_qval
    # F.smooth_l1_loss(qval_a,future_rewards) #alternative loss function if grandients exploded!
    return F.mse_loss(qval_a, future_rewards)


def calc_proj_dist(prob, rewards, dones, vmin, vmax, natoms, dz, gamma):
    """
    Definition: calc_proj_dist(prob,rewards,dones,vmin,vmax,natoms,dz,gamma)

    Parameters
    ----------
    prob : NUMPY ARRAY WITH A SIZE OF (BATCH_SIZE x N ATOMS)
        N ATOMS is usually 51 if.
    rewards : TYPE
        DESCRIPTION.
    dones : TYPE
        DESCRIPTION.
    vmin : TYPE
        DESCRIPTION.
    vmax : TYPE
        DESCRIPTION.
    natoms : TYPE
        DESCRIPTION.
    dz : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    proj : TYPE
        DESCRIPTION.

    """
    proj = np.zeros(prob.shape)
    for atom in range(natoms):
        v = rewards + (vmin + atom * dz) * gamma
        v = np.maximum(vmin, np.minimum(v, vmax))
        idx = (v-vmin)/dz
        l = np.floor(idx).astype(int)
        u = np.ceil(idx).astype(int)
        eq_mask = l == u
        proj[eq_mask, l[eq_mask]] += prob[eq_mask, atom]
        neq_mask = l != u
        proj[neq_mask, l[neq_mask]] += prob[neq_mask, atom] * \
            (idx - l)[neq_mask]
        proj[neq_mask, l[neq_mask]] += prob[neq_mask, atom] * \
            (u - idx)[neq_mask]
    if dones.any():
        proj[dones] = 0.
        d_v = np.maximum(vmin, np.minimum(rewards[dones], vmax))
        d_idx = (d_v-vmin)/dz
        proj[dones, d_idx.astype(int)] = 1.0
    return proj


def calc_dist_loss(batch, net, tgt_net, gamma, vmin, vmax, natoms, dz):
    '''
    Calculate the loss of a categorical DQN batch with C51*N_actions size
    '''
    states, actions, rewards, dones, last_states = unpack_dqn_batch(batch)

    states_v = torch.FloatTensor(states)
    actions_v = torch.tensor(actions)
    last_states_v = torch.FloatTensor(last_states)

    next_dist, next_qvals = tgt_net.target_model.both(last_states_v)
    next_acts = next_qvals.max(dim=1)[1].data.numpy()  # next actions
    next_probs = tgt_net.target_model.softmax(next_dist).data.numpy()
    next_probs_actions = next_probs[range(len(next_acts)), next_acts]

    proj_dist = calc_proj_dist(next_probs_actions, rewards, dones,
                               vmin, vmax, natoms, dz, gamma)
    proj_dist_v = torch.FloatTensor(proj_dist)

    distr_v = net(states_v)
    sa_values = distr_v[range(len(actions_v)), actions_v.data]
    log_sa_values = F.log_softmax(sa_values, dim=1)
    loss = -log_sa_values * proj_dist_v
    return loss.sum(dim=1).mean()


def plot_distributions(prob, proj, i, support, rewards):
    plt.bar(support, prob[i], label='before')
    plt.bar(support, proj[i], label='after')
    plt.text(rewards[i], 0, f'{rewards[i]}')
    plt.legend()


def play_dqn(env, agent):
    s = env.reset()
    rewards = 0.0
    while True:
        env.render()
        s_v = torch.FloatTensor([s])
        a = agent(s_v)[0][0]
        s, r, d, _ = env.step(a)
        rewards += r
        if d:
            print(rewards)
            break
    env.close()
