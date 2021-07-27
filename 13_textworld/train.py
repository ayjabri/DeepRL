#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:39:39 2021

@author: ayman
"""

import argparse
from lib import model, utils, preprocess

PATH = "tw_games"

EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "admissible_commands": True,
}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--prefix', default='treasure', help='prefix of the game')
    parser.add_argument('-s','--seq', default=1,help='Number of games to train with')
    args = parser.parse_args()
    emb_dim, hid_size = 10, 64
    files = [f"tw_games/{args.prefix}{n}.ulx" for n in range(args.seq)]
    env = utils.make_game(files, EXTRA_GAME_INFO)
    env = utils.TextWrapper(env, trainable_info=["inventory","description"])
    preprocessor = preprocess.Preprocessor(env.num_encoders,env.observation_space.vocab_size,
                                   emb_dim, hid_size)
    
    net = model.DQNet(obs_size=env.num_encoders*hid_size, cmd_size=hid_size)
    state = env.reset()
    obs, commands = preprocessor.prep([state])
    
    q_vals = net.q_vals(obs,commands)
    
