#!/usr/bin/env python

import unittest
from game import Narrow
from model import Random
from azero import AlphaZero, Tree


class TestAlphaZero(unittest.TestCase):
    def test_simulate(self):
        game = Narrow()
        model = Random(game)
        az = AlphaZero(game, model)
        state = game.start()
        tree = Tree()
        self.assertTrue(tree.isLeaf())
        value = az.simulate(state, tree)
        self.assertEqual(value, 0)  # Random model always predicts 0 value
        self.assertFalse(tree.isLeaf())
        self.assertEqual(sum(tree.N), 0)  # No child visits
        value = az.simulate(state, tree)
        self.assertEqual(value, -1)  # Reached a game outcome
        self.assertEqual(tree.N.tolist(), [1, 0, 0])  # argmax chooses first
        self.assertEqual(tree.W.tolist(), [-1., 0., 0.])
        self.assertEqual(tree.Q.tolist(), [-1., 0., 0.])
        self.assertEqual(tree.P.tolist(), [1 / 6, 1 / 3, 1 / 3])
        value = az.simulate(state, tree)
        self.assertEqual(value, 0)
        self.assertEqual(tree.N.tolist(), [1, 1, 0])
        self.assertEqual(tree.W.tolist(), [-1., 0., 0.])
        self.assertEqual(tree.Q.tolist(), [-1., 0., 0.])
        self.assertEqual(tree.P.tolist(), [1 / 6, 1 / 6, 1 / 3])
        value = az.simulate(state, tree)
        self.assertEqual(value, 0)
        self.assertEqual(tree.N.tolist(), [1, 1, 1])
        self.assertEqual(tree.W.tolist(), [-1., 0., 0.])
        self.assertEqual(tree.Q.tolist(), [-1., 0., 0.])
        self.assertEqual(tree.P.tolist(), [1 / 6, 1 / 6, 1 / 6])

    def test_select(self):
        game = Narrow()
        model = Random(game)
        az = AlphaZero(game, model)
        state = game.start()
        tree = Tree()
        az.simulate(state, tree)  # first expansion of the tree
        action, leaf = tree.select()
        self.assertTrue(leaf.isLeaf())
        self.assertEqual(action, 0)


if __name__ == '__main__':
    unittest.main()
