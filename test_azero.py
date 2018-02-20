#!/usr/bin/env python

import unittest
import numpy as np
from itertools import product
from game import games, Narrow
from model import models, Uniform
from azero import AlphaZero

N = 10


class TestAlphaZero(unittest.TestCase):
    def test_sample(self):
        for game_cls, model_cls in product(games, models):
            game = game_cls()
            model = model_cls(game.n_action, game.n_view, game.n_player)
            azero = AlphaZero(game, model, sims_per_search=1)
            for _ in range(N):
                state, player, outcome = game.start()
                while outcome is None:
                    action = azero.sample(state, player)
                    state, player, outcome = game.step(state, player, action)

    def check_rank(self, prob, rank):
        assert (-np.sort(-prob) == prob[rank]).all()

    def test_search(self):
        game = Narrow()
        model = Uniform(game.n_action, game.n_view, game.n_player)
        azero = AlphaZero(game, model)
        state, player, _ = game.start()
        probs, _ = azero.search(state, player)
        self.check_rank(probs, [2, 1, 0])
        state, player, _ = game.step(state, player, 2)
        probs, _ = azero.search(state, player)
        self.check_rank(probs, [1, 0, 2])
        state, player, _ = game.step(state, player, 1)
        probs, _ = azero.search(state, player)
        self.check_rank(probs, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
