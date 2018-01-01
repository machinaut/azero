#!/usr/bin/env python

import numpy as np
from azero import Model


class Model:
    ''' Interface for a model used by alphazero '''
    def __init__(self, game):
        ''' Initialize the model (uses game for number of actions, etc) '''
        raise NotImplementedError()

    def model(self, state):
        ''' Return action probability vector and outcome prediction '''
        raise NotImplementedError()

    def update(self, games):
        ''' Update model given games (lists of pairs of probs, outcome)'''
        raise NotImplementedError()


class RandomAgent(Model):
    def __init__(self, game):
        self.n_act = len(game.valid(game.start()))

    def model(self, state):
        return np.ones(self.n_act) / self.n_act, 0

    def update(self, games):
        pass
