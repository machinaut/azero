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


class NearestNeighbor(Model):
    ''' Nearest Neighbor search of training data '''
    def __init__(self, game):
        self.data = {}  # Map from tuple(state) -> (probs, outcome)
        self.n_act = len(game.valid(game.start()))

    def model(self, state):
        ''' Nearest neighbor (L2-distance) result '''
        prior = np.ones(self.n_act) / self.n_act, 0
        best = None
        for tstate, result in self.data.items():
            dist = np.sum(np.square(state - tstate))
            if best is None or dist < best:
                best = dist
                prior = result
        return prior

    def update(self, games):
        ''' Save all most-recent observations per state '''
        for trajectory, outcome in games:
            for state, probs in trajectory:
                self.data[tuple(state)] = (probs, outcome)
