#!/usr/bin/env python

import unittest
from game import Game, games
from util import select


class TestGame(unittest.TestCase):
    def test_random_play(self):
        ''' Try randomly playing all the games '''
        for g in games:
            game = g()
            self.assertIsInstance(game, Game)
            self.check_random_play(game)

    def check_random_play(self, game):
        ''' Randomly play a game many times and check behavior '''
        for _ in range(100):
            state = game.start()
            player = 1
            outcome = None
            self.assertIsNotNone(state)
            while state is not None:
                self.assertIn(player, (1, -1))
                self.assertIsNone(outcome)
                valid = game.valid(state)
                self.assertGreater(sum(valid), 0)
                self.assertLessEqual(sum(valid), len(valid))
                invalid = tuple(not v for v in valid)
                if sum(invalid):
                    inaction = select(invalid)
                    with self.assertRaises(AssertionError):
                        game.step(state, inaction)
                action = select(valid)
                state, player, outcome = game.step(state, action)
            else:
                self.assertIsNone(player)
                self.assertIsNotNone(outcome)
            self.assertIn(outcome, (-1, 0, 1))  # only valid outcomes!


if __name__ == '__main__':
    unittest.main()
