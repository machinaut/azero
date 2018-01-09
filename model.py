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


class Random(Model):
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
        self.n_updates = 0

    def model(self, state):
        ''' Return data if present, else uniform prior '''
        prior = np.ones(self.n_act) / self.n_act, 0
        return self.data.get(tuple(state), prior)

    def update(self, games):
        ''' Save all most-recent observations per state '''
        for trajectory, outcome in games:
            for state, player, probs in trajectory:
                alpha = .1
                old_probs, old_value = self.model(state)
                new_probs = alpha * old_probs + (1 - alpha) * probs
                new_value = alpha * old_value + (1 - alpha) * outcome * player
                self.data[state] = (new_probs, new_value)
        self.n_updates += 1


models = [Random, Memorize]
