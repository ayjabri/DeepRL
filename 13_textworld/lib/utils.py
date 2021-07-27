#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:43:22 2021

@author: ayman
"""
from collections import namedtuple
from textworld.text_utils import extract_vocab_from_gamefiles
from textworld.gym.spaces.text_spaces import Word
from textworld import EnvInfos
from textworld.gym import register_games
import gym
from numpy.random import choice

State = namedtuple(
    'State', ['obs', 'admissible_commands'])


EXT_INFO = {'admissible_commands': True,
            'command_templates': False,
            'description': True,
            'entities': False,
            # 'extras': False,
            'facts': False,
            'feedback': False,
            'game': False,
            'intermediate_reward': True,
            'inventory': True,
            'last_action': False,
            'last_command': False,
            'location': False,
            'lost': False,
            'max_score': False,
            'moves': False,
            'objective': False,
            'policy_commands': False,
            'score': False,
            'verbs': False,
            'won': False
            }


class TextWrapper(gym.Wrapper):
    r"""Wrap TextWorld environment such that it returns consistant observations upon reset and step functions"""

    def __init__(self, env, trainable_info: list = ["inventory"], use_intermediate_rewards=True):
        r"""
        Select from a list of extra attributres to include in training. These attributes should've been 
        requested as extra information when the environment was created.
        """
        super(TextWrapper, self).__init__(env=env)
        self.trainable_info = trainable_info
        self.num_encoders = len(trainable_info) + 1
        self.admissible_commands = []
        self.last_command = []
        self.moves = 0
        self.use_intermediate_rewards = use_intermediate_rewards
        self._init_obs()
        for att in trainable_info:
            self._verify_extra_info(att)

    def _verify_extra_info(self, attribute):
        setattr(self.env.request_infos, attribute, True)

    def _init_obs(self):
        self.extra_info = []
        for att in dir(self.env.request_infos):
            value = getattr(self.env.request_infos, att)
            if isinstance(value, bool) and value:
                self.extra_info.append(att)

    def encode(self, obs, info: dict):
        obs_ = [self.env.observation_space.tokenize(obs)]
        extra_info = {}
        for key, value in info.items():
            if key in self.trainable_info:
                obs_.append(self.env.observation_space.tokenize(value))
            else:
                extra_info[key] = value
        cmds_ = list(map(self.env.action_space.tokenize,
                     info["admissible_commands"]))
        return {"obs":obs_,"admissible_commands":cmds_}, {}

    def reset(self):
        self.moves = 0
        self.admissible_commands = []
        self.last_command = []
        obs, self.last_extra_info = self.env.reset()
        self.last_state = obs
        self.admissible_commands = self.last_extra_info['admissible_commands']
        return self.encode(obs, self.last_extra_info)[0]

    def step(self, action):
        self.moves += 1
        assert action in range(len(self.admissible_commands))
        self.last_command = self.admissible_commands[action]
        _obs, reward, done, self.last_extra_info = self.env.step(self.last_command)
        self.last_state = _obs
        if self.use_intermediate_rewards:
            reward += self.last_extra_info['intermediate_reward']
        self.admissible_commands = self.last_extra_info['admissible_commands']
        state, extra_info = self.encode(_obs, self.last_extra_info)
        return state, reward, done, extra_info
    
    def sample_action(self):
        return choice(range(len(self.admissible_commands)))


def make_game(files: list, extra_info: dict, max_action=8, max_obs=200, max_steps=50):
    vocab = extract_vocab_from_gamefiles(files)
    action_space = Word(max_length=max_action, vocab=vocab)
    observation_space = Word(max_length=max_obs, vocab=vocab)
    env_id = register_games(files, request_infos=EnvInfos(**extra_info), action_space=action_space,
                           observation_space=observation_space, max_episode_steps=max_steps)
    return gym.make(env_id)


filename = ['tw_games/treasure.ulx']
env = make_game(filename, EXT_INFO)

env = TextWrapper(env, trainable_info=["inventory", "description"])



def gen_batch(env, size=10):
    batch = []
    state = env.reset()
    for _ in range(size):
        batch.append(state)
        action = env.sample_action()
        state,r,done,_ = env.step(action)
        if done:
            state = env.reset()
    return batch



def play(env, agent):
    state = env.reset()
    print(env.env.ob[0])
    input()
    rewards = 0
    moves = 0
    while True:
        moves += 1
        action = agent([state])[0][0]
        print(f'### Action:\n\t {env.last_admissible_commands[action]}')
        state, rew, done, _ = env.step(action)
        print(env.unwrapped.obs[0])
        # for key,value in env.last_extra_info.items():
        #     print(key,value)
        input()
        rewards += rew
        if done:
            print(f'Finished within {moves} steps and {rewards} rewards')
            break
