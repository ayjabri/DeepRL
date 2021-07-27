#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:40:13 2021

@author: ayman
"""
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_size):
        super().__init__()
        self.gru = nn.GRU(emb_dim, hid_size, batch_first=True)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, hn = self.gru(x)
        return hn.squeeze(0)


class Preprocessor(nn.Module):
    """Take a batch of tokens with variable lenghts and return fixed length representation"""
    def __init__(self, num_encoders, vocab_size, emb_dim, hid_size):
        r"""
        Args:
        ____
        num_encoders: int, number of trainable information for example: 2 if our state is (observation,inventory)
        vocab_size: int, the size of game vocabulary
        emb_dim: int, embedding dimension that represents each token
        hid_size: hiddden size of memeory layer
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_size = hid_size

        self.emb = nn.Embedding(vocab_size, embedding_dim=emb_dim)

        self.encoders = []
        for i in range(num_encoders):
            enc = Encoder(emb_dim, hid_size)
            self.encoders.append(enc)
            self.add_module(f"enc_{i}", enc)

        self.enc_commands = Encoder(emb_dim, hid_size)
        
    def prep(self, states:list):
        r"""Preprocess an observation from the environment returning a single tensor as an input to DQN network"""
        device = self.emb.weight.device
        obs,cmds = [],[]
        for exp in states:
            obs.append([self.emb(torch.tensor(o).to(device)) for o in exp["obs"]])
            cmds.append([self.emb(torch.tensor(ac).to(device)) for ac in exp["admissible_commands"]])
        return self.encode_sequence(obs), self.encode_commands(cmds)

    def _apply_encoder(self, seq, enc):
        res = rnn_utils.pack_sequence(seq, enforce_sorted=False)
        return enc(res)

    def encode_sequence(self, batch:list):
        r"""Encode a sequence of observation that contains tokenized state. e.g. [[obs_1,extra_info_1]...]"""
        results = []
        for seq, enc in zip(zip(*batch), self.encoders):
            results.append(self._apply_encoder(seq, enc))
        return torch.cat(results, dim=1)

    def encode_commands(self, commands: list):
        cmds = []
        for c in commands:
            cmds.append([self.enc_commands(o.unsqueeze(0)) for o in c])
        return cmds
