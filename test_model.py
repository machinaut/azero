#!/usr/bin/env python

import random
import unittest
import numpy as np
from itertools import product
from model import models, MLP
from game import games, MNOP
from azero import AlphaZero
from nn import loss_fwd
from util import sample_logits, sample_games

N = 10


class TestModel(unittest.TestCase):
    def test_random_play(self):
        for model_cls, game_cls in product(models, games):
            game = game_cls()
            for _ in range(N):
                model = model_cls(game.n_action, game.n_view, game.n_player)
                state, player, outcome = game.start()
                while outcome is None:
                    obs = game.view(state, player)
                    valid = game.valid(state, player)
                    logits, _ = model.model(obs)
                    action = sample_logits(logits, valid)
                    state, player, outcome = game.step(state, player, action)

    def test_nan_propagation(self):
        for model_cls, game_cls in product(models, games):
            game = game_cls()
            if game.n_view == 0:
                continue
            for _ in range(N):
                model = model_cls(game.n_action, game.n_view, game.n_player)
                state, player, outcome = game.start()
                while outcome is None:
                    obs = game.view(state, player)
                    bad_obs = obs.copy().flatten()
                    bad_obs[random.randrange(model.n_obs)] = np.nan
                    bad_obs = bad_obs.reshape(obs.shape)
                    bad_logits, bad_value = model.model(bad_obs)
                    assert np.isnan(bad_value).all()
                    assert np.isnan(bad_logits).all()
                    valid = game.valid(state, player)
                    logits, _ = model.model(obs)
                    action = sample_logits(logits, valid)
                    state, player, outcome = game.step(state, player, action)

    def test_mlp_overfit(self):
        azero = AlphaZero.make(MNOP, MLP, seed=0)
        games = azero.play_multi()
        obs, q, z = sample_games(games, rs=azero.rs)
        loss, _ = azero._model._loss(obs, q, z)
        for i in range(1000):
            last = loss
            azero._model._sparse_update(obs, q, z)
            loss, _ = azero._model._loss(obs, q, z)
            self.assertLess(loss, last)
        true, _ = loss_fwd(np.c_[q, z], q, z, azero._model.c)
        self.assertLess(loss, np.sum(true))


if __name__ == '__main__':
    unittest.main()
