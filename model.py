#!/usr/bin/env python

import numpy as np


class Model:
    ''' Interface for a model used by alphazero '''
    def __init__(self, game):
        ''' Initialize the model (uses game for number of actions, etc) '''
        raise NotImplementedError()

    def model(self, state):
        ''' Return action probability logits and outcome prediction '''
        raise NotImplementedError()

    def update(self, games):
        '''
        Update model given list of games,
        where each game is a list of (state, player, logits)
        '''
        raise NotImplementedError()


class Random(Model):
    ''' The one, the only. '''
    def __init__(self, game):
        self.n_act = len(game.valid(game.start()))

    def model(self, state):
        return np.zeros(self.n_act)

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
        return self.data.get(tuple(state), (np.zeros(self.n_act), 0))

    def update(self, games):
        ''' Save all most-recent observations per state '''
        for trajectory, outcome in games:
            for state, player, logits in trajectory:
                self.data[state] = logits, outcome * player
        self.n_updates += 1


models = [Random, Memorize]
