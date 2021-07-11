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
    parser.add_argument('-g','--game', default='tw_games/treasure.ulx',
            help='Path of game file')
    args = parser.parse_args()
    
    env = utils.make_game(args.game, EXTRA_GAME_INFO)
    env = utils.TextWrapper(env)
    env.reset()

    
