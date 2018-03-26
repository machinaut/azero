#!/usr/bin/env python

import random
import unittest
import numpy as np
from collections import defaultdict
from itertools import product
from util import sample_logits
from game import (games, Game,
                  Null, Binary, Flip, Count, Narrow, Matching, Roshambo,
                  Modulo, Connect3, MNOP, Checkers)

N = 100


class TestGames(unittest.TestCase):
    def test_random_play(self):
        for game_cls in games:
            game = game_cls()
            self.assertIsInstance(game, Game)
            for _ in range(N):
                state, player, outcome = game.start()
                while outcome is None:
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
                        with self.assertRaises(AssertionError, msg='{} {} {}'.format(state, player, action)):
                            game.step(state, player, action)
                # End of game checks
                self.assertIsNone(player)

    def check_trajectory(self, game, traj, out):
        state, player, outcome = game.start()
        for action in traj:
            self.assertIsNone(outcome)
            state, player, outcome = game.step(state, player, action)
        self.assertIsNotNone(outcome)
        np.testing.assert_equal(outcome, out)

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
        self.check_trajectory(Connect3(), (0, 1, 0, 1, 0), (1, -1))
        self.check_trajectory(Connect3(), (0, 1, 0, 2, 4, 3), (-1, 1))
        self.check_trajectory(Connect3(), (0, 1, 1, 2, 3, 2, 2), (1, -1))
        self.check_trajectory(Connect3(), (1, 2, 0, 1, 0, 0), (-1, 1))
        self.check_trajectory(MNOP(), (0, 3, 1, 4, 2), (1, -1))
        self.check_trajectory(MNOP(), (0, 1, 4, 2, 8), (1, -1))
        self.check_trajectory(MNOP(), (0, 1, 3, 2, 6), (1, -1))
        self.check_trajectory(MNOP(), (2, 1, 4, 0, 6), (1, -1))
        self.check_trajectory(MNOP(), (3, 0, 4, 1, 6, 2), (-1, 1))
        self.check_trajectory(MNOP(), (3, 6, 1, 4, 5, 2), (-1, 1))
        self.check_trajectory(MNOP(2, 2, 2), (0, 1, 2), (1, -1))
        self.check_trajectory(MNOP(2, 2, 2), (0, 1, 3), (1, -1))
        self.check_trajectory(MNOP(2, 2, 2), (0, 2, 1), (1, -1))
        self.check_trajectory(MNOP(2, 2, 2), (1, 0, 2), (1, -1))
        with self.assertRaises(AssertionError):
            self.check_trajectory(MNOP(), (0, 0), None)
        with self.assertRaises(AssertionError):
            self.check_trajectory(MNOP(), (0, 1, 1), None)

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
                    view = game.view(state, player).tostring()
                    tstate = tuple(state)
                    state_valid_view.append((tstate, valid, view))
                    state_player.append((tstate, player))
                    action = sample_logits((0,) * len(valid), valid)
                    state, player, outcome = game.step(state, player, action)
            self.check_conditional_independence(state_valid_view)
            self.check_dependence(state_player)
    
    def test_checkers(self):
        game = Checkers()
        #import ipdb; ipdb.set_trace()
        self.assertEqual(game.jumps_fwd[0],[(8,2,5)])
        self.assertEqual(game.jumps_fwd[1],[(1,2,4)])
        self.assertEqual(game.jumps_fwd[2],[(10,5,7)])
        self.assertEqual(game.jumps_fwd[3],[(3,5,6)])
        self.assertEqual(game.jumps_fwd[4],[])
        self.assertEqual(game.jumps_fwd[5],[])
        self.assertEqual(game.jumps_fwd[6],[])
        self.assertEqual(game.jumps_fwd[7],[])
        self.assertEqual(game.moves_fwd[0],[(8,2)])
        self.assertEqual(game.moves_fwd[1],[(9,3),(1,2)])
        self.assertEqual(game.moves_fwd[2],[(2,4),(10,5)])
        self.assertEqual(game.moves_fwd[3],[(3,5)])
        self.assertEqual(game.moves_fwd[4],[(12,6)])
        self.assertEqual(game.moves_fwd[5],[(13,7),(5,6)])
        self.assertEqual(game.moves_fwd[6],[])
        self.assertEqual(game.moves_fwd[7],[])
        self.assertEqual(game.moves_bak[0],[])
        self.assertEqual(game.moves_bak[1],[])
        self.assertEqual(game.moves_bak[2],[(18,0),(26,1)])
        self.assertEqual(game.moves_bak[3],[(19,1)])
        self.assertEqual(game.moves_bak[4],[(28,2)])
        self.assertEqual(game.moves_bak[5],[(29,3),(21,2)])
        self.assertEqual(game.moves_bak[6],[(22,4),(30,5)])
        self.assertEqual(game.moves_bak[7],[(23,5)])
        self.assertEqual(game.jumps_bak[0],[])
        self.assertEqual(game.jumps_bak[1],[])
        self.assertEqual(game.jumps_bak[2],[])
        self.assertEqual(game.jumps_bak[3],[])
        self.assertEqual(game.jumps_bak[4],[(28,2,1)])
        self.assertEqual(game.jumps_bak[5],[(21,2,0)])
        self.assertEqual(game.jumps_bak[6],[(30,5,3)])
        self.assertEqual(game.jumps_bak[7],[(23,5,2)])


if __name__ == '__main__':
    unittest.main()
