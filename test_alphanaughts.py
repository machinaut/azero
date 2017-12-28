#!/usr/bin/env python

import unittest

from alphanaughts import AlphaNaughts


class TestNaughts(unittest.TestCase):
    def test_three(self):
        an = AlphaNaughts()
        threes = [set([(1, 2), (3, 6), (4, 8)]),
                  set([(0, 2), (4, 7)]),
                  set([(0, 1), (4, 6), (5, 8)]),
                  set([(0, 6), (4, 5)]),
                  set([(1, 7), (3, 5), (0, 8), (2, 6)]),
                  set([(2, 8), (3, 4)]),
                  set([(0, 3), (2, 4), (7, 8)]),
                  set([(1, 4), (6, 8)]),
                  set([(0, 4), (2, 5), (6, 7)]),
                  ]
        for move, three in enumerate(threes):
            res = set((min(a, b), max(a, b)) for a, b in an.three(move))
            self.assertEqual(res, three)

    def test_step(self):
        an = AlphaNaughts()
        state = AlphaNaughts.START
        for i in range(9):
            res = tuple(-1 if i == j else 0 for j in range(9))
            self.assertEqual(an.step(state, i), (res, None))

    def test_win(self):
        an = AlphaNaughts()
        wins = [((0, 1, 1, 0, 0, 0, 0, 0, 0), 0),
                ((1, 0, 1, 0, 0, 0, 0, 0, 0), 1),
                ((1, 1, 0, 0, 0, 0, 0, 0, 0), 2),
                ((1, 0, 0, 0, 0, 0, 1, 0, 0), 3),
                ((0, 1, 0, 0, 0, 0, 0, 1, 0), 4),
                ((0, 0, 1, 0, 0, 0, 0, 0, 1), 5),
                ((0, 0, 1, 0, 1, 0, 0, 0, 0), 6),
                ((0, 1, 0, 0, 1, 0, 0, 0, 0), 7),
                ((1, 0, 0, 0, 1, 0, 0, 0, 0), 8),
                ((1, 0, 0, 0, 0, 0, 0, 0, 1), 4),
                ((0, 0, 1, 0, 0, 1, 0, 0, 0), 8)]
        for state, move in wins:
            self.assertEqual(an.step(state, move), (None, +1))

    def test_draw(self):
        an = AlphaNaughts()
        draw = [((1, -1, 1, 1, -1, -1, -1, 1, 0), 8),
                ((0, -1, 1, -1, -1, 1, -1, 1, 1), 0),
                ((-1, 0, -1, 1, -1, 1, -1, 1, -1), 1),
                ((-1, 1, -1, 1, -1, 1, -1, 0, -1), 7)]
        for state, move in draw:
            self.assertEqual(an.step(state, move), (None, 0))


if __name__ == '__main__':
    unittest.main()
