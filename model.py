#!/usr/bin/env python

import numpy as np
import nn
from util import pairwise


class Model:
    ''' Interface class for a model to be optimized by alphazero algorithm '''
    def __init__(self, n_action, n_view, n_player):
        self.n_act = n_action
        self.n_obs = n_view + 1  # Includes player
        self.n_val = n_player
        self.n_updates = 0

    def model(self, obs):
        '''
        Call the model on a board state
            obs - game state concatenated with current player
        Returns
            logits - action selection probability logits (pre-softmax)
            values - estimated sum of future rewards per player
        '''
        assert obs.shape == (self.n_obs,)
        logits, values = self._model(obs)
        assert logits.shape == (self.n_act,)
        assert values.shape == (self.n_val,)
        return logits, values

    def _model(self, obs):
        raise NotImplementedError('Implement in subclass')

    def update(self, games):
        '''
        Update model given a list of games.  Each game is a pair of:
            trajectory - list of (obs, logits)
            outcome - total reward per player
        '''
        self.n_updates += 1
        self._update(games)

    def _update(self, games):
        raise NotImplementedError()


class Uniform(Model):
    ''' Maximum entropy (uniform distribution) '''
    def _model(self, obs):
        # Fun little hack, sum the observation, then multiply by zero
        # This allows NaN propagation, which is a great way of testing models
        zero = obs.sum() * 0.0
        logits = np.ones(self.n_act) * zero
        values = np.ones(self.n_val) * zero
        return logits, values


class Linear(Model):
    ''' Simple linear model '''
    def __init__(self, *args, seed=None, scale=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        rs = np.random.RandomState(seed)
        self.W = rs.randn(self.n_obs, self.n_act) * scale
        self.V = rs.randn(self.n_obs, self.n_val) * scale

    def _model(self, obs):
        logits = obs.dot(self.W)
        values = obs.dot(self.V)
        return logits, values


class Memorize(Model):
    ''' Remember and re-use training data '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {}  # Map from tuple(state) -> (logits, outcome)

    def _model(self, obs):
        ''' Return data if present, else uniform prior '''
        # Hack to ensure NaN propagation
        zero = np.sum(obs) * 0.0
        logits = np.ones(self.n_act) * zero
        values = np.ones(self.n_val) * zero
        return self.data.get(tuple(obs), (logits, values))


class MLP(Model):
    def __init__(self, *args, seed=None, scale=0.01, hidden_dims=[100], **kwargs):
        super().__init__(*args, **kwargs)
        rs = np.random.RandomState(seed)
        self.params = dict()
        all_dims = [self.n_obs] + hidden_dims + [self.n_act + self.n_val]
        for i, (in_dim, out_dim) in enumerate(pairwise(all_dims)):
            self.params['W%d' % i] = rs.randn(in_dim, out_dim) * scale
            self.params['b%d' % i] = np.zeros(out_dim)
        self.n_layer = len(all_dims) - 1

    def _model(self, x):
        for i in range(self.n_layer):
            x, _ = nn.mlp_fwd(x, self.params['W%d' % i], self.params['b%d' % i])
            if i < self.n_layer - 1:
                x, _ = nn.relu_fwd(x)
        return x[:self.n_act], x[self.n_act:]


models = [Uniform, Linear, Memorize, MLP]
