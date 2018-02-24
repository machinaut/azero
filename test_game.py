#!/usr/bin/env python

import random
import unittest
from collections import defaultdict
from itertools import product
from util import sample_logits
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

    def check_conditional_independence(self, data):
        ''' Ensure conditional independence of X and Y given Z '''
        X = defaultdict(int)
        Y = defaultdict(int)
        Z = defaultdict(int)
        XZ = defaultdict(int)
        YZ = defaultdict(int)
        XYZ = defaultdict(int)
        for x, y, z in data:
            X[x] += 1
            Y[y] += 1
            Z[z] += 1
            XZ[(x, z)] += 1
            YZ[(y, z)] += 1
            XYZ[(x, y, z)] += 1
        for x, y, z in XYZ.keys():
            self.assertEqual(XYZ[(x, y, z)] * Z[z], XZ[(x, z)] * YZ[(y, z)])
        for x, (y, z) in product(X.keys(), YZ.keys()):
            if (x, y, z) not in XYZ:
                assert (x, z) not in XZ or (y, z) not in YZ
        for y, (x, z) in product(Y.keys(), XZ.keys()):
            if (x, y, z) not in XYZ:
                assert (x, z) not in XZ or (y, z) not in YZ

    def check_dependence(self, data):
        ''' Ensure that Y depends entirely on X '''
        X = dict()
        for x, y in data:
            if x not in X:
                X[x] = y
            self.assertEqual(X[x], y)

    def test_statistics(self):
        '''
        Statistical Quantity tests
            - Ensure valid doesn't give extra information about state
            - Ensure that player wholly depends on state
        '''
        for game_cls in games:
            game = game_cls()
            state_valid_view = []
            state_player = []
            for _ in range(N):
                state, player, outcome = game.start()
                while outcome is None:
                    valid = game.valid(state, player)
                    view = game.view(state, player)
                    state_valid_view.append((state, valid, view))
                    state_player.append((state, player))
                    action = sample_logits((0,) * len(valid), valid)
                    state, player, outcome = game.step(state, player, action)
            self.check_conditional_independence(state_valid_view)
            self.check_dependence(state_player)


if __name__ == '__main__':
    unittest.main()
