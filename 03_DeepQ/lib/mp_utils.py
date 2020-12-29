# -*- coding: utf-8 -*-

import gym
import ptan
from collections import namedtuple

EpisodeEnd = namedtuple('EpisodeEnd', ['step', 'reward', 'epsilon'])


def data_fun(net, exp_queue, params):
    """
    Definition: data_fun(net,exp_queue,ENV_ID,STEPS=1,N_ENVS=1)

    Stores ptan FirstLast experiences in a multiprocess Queue()

    Parameters
    ----------
    net : Deep-Q Neural Netwok class
        Can be any DQN. Tested with DuelDQN network

    exp_queue : Pytorch Multiprocessing.Queue()
        Shared Queue to store experiences.

    params : a simple name space dict that contains hyperparameters

    Returns
    -------
    Stores experiences in a multiprocessing Queue(). It also stores step,reward and epsilon
    as named tuple (EndEpisode) at the end of each episode.

    Use as target for Multiprocessing.  

    N-Environments:
    --------
    To use N number of environments you must do the following changes to your training loop:
        1- Use common SEED in all environments        

        2- Multiply batch-size by N        

        3- Multipy frame by N in Epsilon tracker.frame() function if using one      

        4- Multiply fps by N (haven't tried it yet!)       

        5- Populate N steps if using Buffer       
    """
    envs = []
    for _ in range(params.N_ENVS):
        env = gym.make(params.ENV_ID)
        env.seed(params.SEED)
        envs.append(env)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(
        net, selector, preprocessor=ptan.agent.float32_preprocessor)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.EPS_START, params.EPS_END,
                                              params.EPS_FRAMES)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent,
                                                           params.GAMMA, steps_count=params.STEPS)
    step = 0
    for exp in exp_source:
        step += 1
        eps_tracker.frame(step*params.N_ENVS)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            exp_queue.put(EpisodeEnd(step, new_reward[0], selector.epsilon))
        exp_queue.put(exp)


def c51_data_fun(net, exp_queue, params):
    """
    Definition: c51_data_fun(net,exp_queue,ENV_ID,STEPS=1,N_ENVS=1)

    Stores ptan FirstLast experiences in a multiprocess Queue() for Categorical DQN (C51)

    Parameters
    ----------
    net : Deep-Q Neural Netwok class
        Can be any DQN. Tested with DuelDQN network

    exp_queue : Pytorch Multiprocessing.Queue()
        Shared Queue to store experiences.

    params : a simple name space dict that contains hyperparameters

    Returns
    -------
    Stores experiences in a multiprocessing Queue(). It also stores step,reward and epsilon
    as named tuple (EndEpisode) at the end of each episode.

    Use as target for Multiprocessing.  

    N-Environments:
    --------
    To use N number of environments you must do the following changes to your training loop:
        1- Use common SEED in all environments        

        2- Multiply batch-size by N        

        3- Multipy frame by N in Epsilon tracker.frame() function if using one      

        4- Multiply fps by N (haven't tried it yet!)       

        5- Populate N steps if using Buffer       
    """
    envs = []
    for _ in range(params.N_ENVS):
        env = gym.make(params.ENV_ID)
        env.seed(params.SEED)
        envs.append(env)
    selector = ptan.actions.EpsilonGreedyActionSelector()
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(
        x), selector, preprocessor=ptan.agent.float32_preprocessor)
    eps_tracker = ptan.actions.EpsilonTracker(selector, params.EPS_START, params.EPS_END,
                                              params.EPS_FRAMES)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent,
                                                           params.GAMMA, steps_count=params.STEPS)
    step = 0
    for exp in exp_source:
        step += 1
        eps_tracker.frame(step*params.N_ENVS)
        new_reward = exp_source.pop_total_rewards()
        if new_reward:
            exp_queue.put(EpisodeEnd(step, new_reward[0], selector.epsilon))
        exp_queue.put(exp)


class MPBatchGenerator(object):
    """
    Yields batchs from experiences stored in multiprocess Queue()


    Parameters:
    -------
    buffer: ptan.experience.ExperienceReplayBuffer(exp_source=None)
        Buffer object that will store FirstLast experiences

    exp_queue: Torch Multiprocessing Queue()
        Queue of specific size the will store observations and end of episode readings

    initial: Int
        Number of stored experiences before start sampling

    batch_size: int
        The size of batch to generate

    multiplier: int. Defaults to 1
        Multiply batch size by this number
    """

    def __init__(self, buffer, exp_queue, initial, batch_size, multiplier):
        self.buffer = buffer
        self.exp_queue = exp_queue
        self.initial = initial
        self.batch_size = batch_size
        self.multiplier = multiplier
        self._total_rewards = []
        self.frame = 0
        self.episode = 0
        self.epsilon = 0.0

    def pop_rewards_idx_eps(self):
        res = list(self._total_rewards)
        self._total_rewards.clear()
        return res

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        while True:
            while not self.exp_queue.empty():
                exp = self.exp_queue.get()
                if isinstance(exp, EpisodeEnd):
                    self._total_rewards.append(exp.reward)
                    self.frame = exp.step
                    self.epsilon = exp.epsilon
                    self.episode += 1
                else:
                    self.buffer._add(exp)
                    self.frame += 1
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size * self.multiplier)
