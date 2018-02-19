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


class Model:
    ''' Interface class for a model to be optimized by alphazero algorithm '''
    def __init__(self, n_action: int, n_state: int) -> None:
        self.n_act = n_action
        self.n_obs = n_state + 1  # Includes player

    def model(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
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
        return logits, value

    def _model(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError('Implement in subclass')


class Uniform(Model):
    def _model(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        # Fun little hack, sum the observation, then multiply by zero
        # This allows NaN propagation, which is a great way of testing models
        zero = cast(float, obs.sum()) * 0.0
        logits = np.ones(self.n_act, dtype=float) * zero
        return logits, zero


class Linear(Model):
    def __init__(self, n_action: int, n_state: int,
                 seed: Optional[int] = None,
                 weight_scale: float = 0.01) -> None:
        super().__init__(n_action, n_state)
        rs = np.random.RandomState(seed)
        self.W = rs.randn(self.n_obs, self.n_act) * weight_scale
        self.V = rs.randn(self.n_obs) * weight_scale

    def _model(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        logits = obs.dot(self.W)
        value = cast(float, obs.dot(self.V))  # ugh hack
        return logits, value


models = [Uniform]


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
