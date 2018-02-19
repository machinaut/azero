#!/usr/bin/env python

import unittest
from itertools import product
from game import games
from model import models
from azero import AlphaZero

N = 10


class TestAlphaZero(unittest.TestCase):
    def test_sample(self):
        for game_cls, model_cls in product(games, models):
            game = game_cls()
            model = model_cls(game.n_action, game.n_view)
            azero = AlphaZero(game, model)
            for _ in range(N):
                state, player, outcome = game.start()
                while outcome is not None:
                    action = azero.sample(state, player)
                    state, player, outcome = game.step(state, player, action)


if __name__ == '__main__':
    unittest.main()
