#!/usr/bin/env python

import random
import unittest
import numpy as np
from itertools import product
from model import models
from game import games
from util import view2obs, sample_logits

N = 10


class TestModel(unittest.TestCase):
    def test_random_play(self):
        for model_cls, game_cls in product(models, games):
            game = game_cls()
            for _ in range(N):
                model = model_cls(game.n_action, game.n_view, game.n_player)
                state, player, outcome = game.start()
                while outcome is None:
                    view = game.view(state, player)
                    obs = view2obs(view, player)
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
                    view = game.view(state, player)
                    obs = view2obs(view, player)
                    bad_obs = obs.copy()
                    bad_obs[random.randrange(len(bad_obs))] = np.nan
                    bad_logits, bad_value = model.model(bad_obs)
                    assert np.isnan(bad_value).all()
                    assert np.isnan(bad_logits).all()
                    valid = game.valid(state, player)
                    logits, _ = model.model(obs)
                    action = sample_logits(logits, valid)
                    state, player, outcome = game.step(state, player, action)


if __name__ == '__main__':
    unittest.main()
