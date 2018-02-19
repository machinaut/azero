#!/usr/bin/env python

import numpy as np


class Model:
    ''' Interface class for a model to be optimized by alphazero algorithm '''
    def __init__(self, n_action, n_view):
        self.n_act = n_action
        self.n_obs = n_view + 1  # Includes player
        self.n_updates = 0

    def model(self, obs):
        '''
        Call the model on a board state
            obs - game state concatenated with current player
        Returns
            logits - action selection probability logits (pre-softmax)
            value - estimated value of the board state to this player
        '''
        assert obs.shape == (self.n_obs,)
        logits, value = self._model(obs)
        assert logits.shape == (self.n_act,)
        assert isinstance(value, float)
        return logits, value

    def _model(self, obs):
        raise NotImplementedError('Implement in subclass')

    def update(self, games):
        '''
        Update model given list of games,
        where each game is a list of (state, player, logits)
        '''
        self.n_updates += 1
        self._update(games)

    def _update(self, games):
        raise NotImplementedError()


class Uniform(Model):
    def _model(self, obs):
        # Fun little hack, sum the observation, then multiply by zero
        # This allows NaN propagation, which is a great way of testing models
        zero = obs.sum() * 0.0
        logits = np.ones(self.n_act, dtype=float) * zero
        return logits, zero


class Linear(Model):
    def __init__(self, n_action, n_view, seed=None, weight_scale=0.01):
        super().__init__(n_action, n_view)
        rs = np.random.RandomState(seed)
        self.W = rs.randn(self.n_obs, self.n_act) * weight_scale
        self.V = rs.randn(self.n_obs) * weight_scale

    def _model(self, obs):
        logits = obs.dot(self.W)
        value = obs.dot(self.V)
        return logits, value


class Memorize(Model):
    ''' Remember and re-use training data '''
    def __init__(self, n_action, n_view):
        super().__init__(n_action, n_view)
        self.data = {}  # Map from tuple(state) -> (logits, outcome)

    def _model(self, obs):
        ''' Return data if present, else uniform prior '''
        # Hack to ensure NaN propagation
        zero = np.sum(obs) * 0.0
        zeros = np.ones(self.n_act) * zero
        return self.data.get(tuple(obs), (zeros, zero))

    def _update(self, games):
        ''' Save all most-recent observations per state '''
        for trajectory, outcome in games:
            for state, player, logits in trajectory:
                self.data[state] = logits, outcome * player


models = [Uniform, Linear, Memorize]
