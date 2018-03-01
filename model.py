#!/usr/bin/env python

import numpy as np
from collections import OrderedDict

import nn
from util import pairwise, sample_games


class Model:
    ''' Interface class for a model to be optimized by alphazero algorithm '''
    def __init__(self, n_action, n_view, n_player, seed=None):
        self.rs = np.random.RandomState(seed=seed)
        self.n_act = n_action
        self.n_obs = n_view
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
        obs = np.asarray(obs, dtype=float)
        assert obs.size == self.n_obs
        logits, values = self._model(obs)
        logits = np.asarray(logits, dtype=float)
        values = np.asarray(values, dtype=float)
        assert logits.size == self.n_act
        assert values.size == self.n_val
        return logits, values

    def _model(self, obs):
        raise NotImplementedError('Implement in subclass')

    def update(self, games):
        '''
        Update model given a list of games.  Each game is a pair of:
            trajectory - list of (obs, probs)
            outcome - total reward per player
        '''
        self.n_updates += 1
        self._update(games)

    def _update(self, games):
        # Optionally overwrite this to get dense updates
        # Default is to sample single data point from each game
        # and pass them to _sparse_update().
        obs, q, z = sample_games(games)
        self._sparse_update(obs, q, z)

    def _sparse_update(self, obs, q, z):
        raise NotImplementedError('Implement this or _update() in subclass')


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
    def __init__(self, *args, scale=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = self.rs.randn(self.n_obs, self.n_act) * scale
        self.V = self.rs.randn(self.n_obs, self.n_val) * scale

    def _model(self, obs):
        obs = obs.flatten()
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
        return self.data.get(obs.tostring(), (logits, values))


class MLP(Model):
    def __init__(self, *args, scale=1e-4, hidden_dims=[10], c=0.5,
                 learning_rate=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c  # Linear combination of loss terms (probs and values)
        self.learning_rate = learning_rate  # Step size of updates
        self.params = OrderedDict()
        all_dims = [self.n_obs] + hidden_dims + [self.n_act + self.n_val]
        for i, (in_dim, out_dim) in enumerate(pairwise(all_dims)):
            self.params['W%d' % i] = self.rs.randn(in_dim, out_dim) * scale
            self.params['b%d' % i] = np.zeros(out_dim)
        self.n_layer = len(all_dims) - 1

    def _model(self, obs):
        x, _ = self._fwd(obs.reshape(1, -1))
        return x[0, :self.n_act], x[0, self.n_act:]

    def _fwd(self, x):
        cache = OrderedDict()
        for i in range(self.n_layer):
            W, b = self.params['W%d' % i], self.params['b%d' % i]
            x, cache[i] = nn.mlp_fwd(x.reshape(x.shape[0], -1), W, b)
            if i < self.n_layer - 1:
                x, cache['r%d' % i] = nn.relu_fwd(x)
        return x, cache

    def _bak(self, dx, cache):
        grads = OrderedDict()
        for i in range(self.n_layer - 1, -1, -1):
            if i < self.n_layer - 1:
                dx = nn.relu_bak(dx, cache['r%d' % i])
            dx, grads['W%d' % i], grads['b%d' % i] = nn.mlp_bak(dx, cache[i])
        return grads

    def _loss(self, obs, q, outcome):
        obs = obs.reshape(obs.shape[0], -1)
        x, cache = self._fwd(obs)
        loss, cache['loss'] = nn.loss_fwd(x, q, outcome, self.c)
        dx = nn.loss_bak(np.ones(1), cache['loss'])
        grads = self._bak(dx, cache)
        return np.sum(loss), grads

    def _sparse_update(self, x, q, z):
        loss, grads = self._loss(x, q, z)
        for name in self.params.keys():
            self.params[name] -= self.learning_rate * grads[name]


models = [Uniform, Linear, Memorize, MLP]
