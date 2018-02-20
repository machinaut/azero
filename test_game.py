#!/usr/bin/env python

import random
import unittest
from game import (games, Game,
                  Null, Binary, Flip, Count, Narrow, Matching, Roshambo,
                  Modulo)

N = 100


class TestGames(unittest.TestCase):
    def test_random_play(self):
        for game_cls in games:
            game = game_cls()
            self.assertIsInstance(game, Game)
            for _ in range(N):
                state, player, outcome = game.start()
                while outcome is None:
                    # Verify incorrect length state asserts valid()
                    if len(state) > 1:
                        with self.assertRaises(AssertionError):
                            game.valid(state[:-1], player)
                    with self.assertRaises(AssertionError):
                        game.valid(state + (0,), player)
                    # Verify incorrect length state asserts view()
                    if len(state) > 1:
                        with self.assertRaises(AssertionError):
                            game.view(state[:-1], player)
                    with self.assertRaises(AssertionError):
                        game.view(state + (0,), player)
                    # Verify incorrect player asserts valid()
                    with self.assertRaises(AssertionError):
                        game.valid(state, player + 1)
                    with self.assertRaises(AssertionError):
                        game.valid(state, player - 1)
                    # Verify incorrect player asserts view()
                    with self.assertRaises(AssertionError):
                        game.view(state, player + 1)
                    with self.assertRaises(AssertionError):
                        game.view(state, player - 1)
                    # Get a view for the current player
                    game.view(state, player)
                    # Get a valid mask for the current state
                    valid = game.valid(state, player)
                    # Check that at least one action is valid
                    self.assertGreater(sum(valid), 0)
                    # Draw a random action, maybe invalid
                    action = random.randrange(game.n_action)
                    # Verify that incorrect length state asserts step()
                    if len(state) > 1:
                        with self.assertRaises(AssertionError):
                            game.step(state[:-1], player, action)
                    with self.assertRaises(AssertionError):
                        game.step(state + (0,), player, action)
                    # Verify that out-of-range action asserts step()
                    with self.assertRaises(AssertionError):
                        game.step(state, player, -1)
                    with self.assertRaises(AssertionError):
                        game.step(state, player, len(valid))
                    # Verify that improper player asserts step()
                    with self.assertRaises(AssertionError):
                        game.step(state, player + 1, action)
                    with self.assertRaises(AssertionError):
                        game.step(state, player - 1, action)
                    # Take the action, if invalid, raise assert
                    if valid[action]:
                        state, player, outcome = game.step(state, player, action)
                    else:  # Verify that invalid action raises assert
                        with self.assertRaises(AssertionError):
                            game.step(state, player, action)
                # End of game checks
                self.assertIsNone(player)
                self.assertIsNone(state)

    def check_trajectory(self, game, traj, out):
        state, player, outcome = game.start()
        for action in traj:
            self.assertIsNone(outcome)
            state, player, outcome = game.step(state, player, action)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome, out)

    def test_trajectories(self):
        self.check_trajectory(Null(), (), (0,))
        self.check_trajectory(Binary(), (0,), (-1,))
        self.check_trajectory(Binary(), (1,), (1,))
        with self.assertRaises(AssertionError):
            self.check_trajectory(Binary(), (2,), None)
        self.check_trajectory(Flip(0), (0,), (-1,))
        self.check_trajectory(Flip(0), (1,), (1,))
        self.check_trajectory(Flip(1), (0,), (1,))
        with self.assertRaises(AssertionError):
            self.check_trajectory(Flip(0), (2,), None)
        self.check_trajectory(Count(), (0, 1, 2), (1,))
        self.check_trajectory(Count(), (0, 1, 1), (-1,))
        self.check_trajectory(Count(), (0, 1, 0), (-1,))
        self.check_trajectory(Count(), (1,), (-1,))
        self.check_trajectory(Count(), (2,), (-1,))
        with self.assertRaises(AssertionError):
            self.check_trajectory(Count(), (3,), None)
        self.check_trajectory(Narrow(), (0,), (-1,))
        self.check_trajectory(Narrow(), (1, 0), (-1,))
        self.check_trajectory(Narrow(), (2, 1, 0), (-1,))
        self.check_trajectory(Narrow(), (2, 0), (-1,))
        with self.assertRaises(AssertionError):
            self.check_trajectory(Narrow(), (1, 1), None)
        with self.assertRaises(AssertionError):
            self.check_trajectory(Narrow(), (2, 2), None)
        with self.assertRaises(AssertionError):
            self.check_trajectory(Narrow(), (3,), None)
        self.check_trajectory(Matching(), (0, 0), (-1, 1))
        self.check_trajectory(Matching(), (0, 1), (1, -1))
        self.check_trajectory(Matching(), (1, 0), (1, -1))
        self.check_trajectory(Matching(), (1, 1), (-1, 1))
        with self.assertRaises(AssertionError):
            self.check_trajectory(Matching(), (2,), None)
        self.check_trajectory(Roshambo(), (0, 0), (-1, -1))
        self.check_trajectory(Roshambo(), (0, 1), (1, -1))
        self.check_trajectory(Roshambo(), (0, 2), (-1, 1))
        self.check_trajectory(Roshambo(), (1, 0), (-1, 1))
        self.check_trajectory(Roshambo(), (1, 1), (-1, -1))
        self.check_trajectory(Roshambo(), (1, 2), (1, -1))
        self.check_trajectory(Roshambo(), (2, 0), (1, -1))
        self.check_trajectory(Roshambo(), (2, 1), (-1, 1))
        self.check_trajectory(Roshambo(), (2, 2), (-1, -1))
        with self.assertRaises(AssertionError):
            self.check_trajectory(Roshambo(), (3,), None)
        self.check_trajectory(Modulo(), (0, 0, 0), (1, -1, -1))
        self.check_trajectory(Modulo(), (0, 1, 0), (-1, 1, -1))
        self.check_trajectory(Modulo(), (1, 0, 1), (-1, -1, 1))
        self.check_trajectory(Modulo(), (2, 2, 2), (1, -1, -1))
        self.check_trajectory(Modulo(), (2, 1, 1), (-1, 1, -1))
        with self.assertRaises(AssertionError):
            self.check_trajectory(Modulo(), (3,), None)


if __name__ == '__main__':
    unittest.main()
