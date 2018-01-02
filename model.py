#!/usr/bin/env python

import numpy as np


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
    ''' The one, the only. '''
    def __init__(self, game):
        self.n_act = len(game.valid(game.start()))

    def model(self, state):
        return np.ones(self.n_act) / self.n_act, 0

    def update(self, games):
        pass


class Memorize(Model):
    ''' Remember and re-use training data '''
    def __init__(self, game):
        self.data = {}  # Map from tuple(state) -> (probs, outcome)
        self.n_act = len(game.valid(game.start()))

    def model(self, state):
        ''' Return data if present, else uniform prior '''
        prior = np.ones(self.n_act) / self.n_act, 0
        return self.data.get(tuple(state), prior)

    def update(self, games):
        ''' Save all most-recent observations per state '''
        for trajectory, outcome in games:
            for state, probs in trajectory:
                self.data[tuple(state)] = (probs, outcome)
