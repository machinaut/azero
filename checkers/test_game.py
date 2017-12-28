#!/usr/bin/env python

import unittest
from game import square2rank, BoardState


class TestMisc(unittest.TestCase):
    def test_square2rank(self):
        self.assertEqual(square2rank(1), 1)
        self.assertEqual(square2rank(4), 1)
        self.assertEqual(square2rank(5), 2)
        self.assertEqual(square2rank(19), 5)
        self.assertEqual(square2rank(32), 8)


class TestGame(unittest.TestCase):
    def test_valid(self):
        bs = BoardState()
        self.assertTrue(bs.valid(9, 14))


if __name__ == '__main__':
    unittest.main()
