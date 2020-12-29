#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 05:43:13 2020

@author: ayman
"""
from types import SimpleNamespace

HYPERPARAMS = {'lander': SimpleNamespace(**
                                         {'env_id': 'LunarLanderContinuous-v2',
                                          'n_envs': 8,
                                          'seed': 120,
                                          'eps_start': 1.0,
                                          'eps_end': 0.02,
                                          'eps_frames': 25_000,
                                          'obs_size': 8,
                                          'act_size': 2,
                                          'high_action': 1,
                                          'hid_size': 128,
                                          'gamma': 0.99,
                                          'steps': 4,
                                          'buffer_size': 20_000,
                                          'init_replay': 2_000,
                                          'lr': 4e-4,
                                          'n_mp': 4,
                                          'batch_size': 32,
                                          'vmin': -500,
                                          'vmax': 300,
                                          'natoms': 51,
                                          'bound_solve': 200,
                                          'sync_net': 1000,
                                          'skip': None
                                          }),
               'pendulum': SimpleNamespace(**
                                           {'env_id': 'Pendulum-v0',
                                            'n_envs': 2,
                                            'seed': 144,
                                            'eps_start': 1.0,
                                            'eps_end': 0.02,
                                            'eps_frames': 5_000,
                                            'obs_size': 3,
                                            'act_size': 1,
                                            'high_action': 2,
                                            'hid_size': 64,
                                            'gamma': 0.99,
                                            'steps': 5,
                                            'buffer_size': 15_000,
                                            'init_replay': 1_500,
                                            'lr': 1e-4,
                                            'n_mp': 4,
                                            'batch_size': 32,
                                            'vmin': -1000,
                                            'vmax': 0,
                                            'natoms': 51,
                                            'bound_solve': 0,
                                            'sync_net': 1000,
                                            'skip': None
                                            }),

               }
