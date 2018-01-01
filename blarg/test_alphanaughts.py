#!/usr/bin/env python

import unittest
import numpy as np

from alphanaughts import AlphaNaughts


class TestNaughts(unittest.TestCase):
    def test_step(self):
        an = AlphaNaughts()
        state = AlphaNaughts.START
        for i in range(9):
            res = tuple(-1 if i == j else 0 for j in range(9))
            self.assertEqual(an.step(state, i), (res, None))

    def test_random_valid(self):
        an = AlphaNaughts()
        for _ in range(100):
            state = tuple(np.random.choice((-1, 0, 1), 9))
            if an.done(state) is None:
                move = an.random(state)
                self.assertTrue(an.valid(state)[move])

    def test_step_random(self):
        an = AlphaNaughts()
        for _ in range(100):  # Number of random games to play
            state = an.START
            count = 0
            while state is not None:
                self.assertIsInstance(state, tuple)
                move = an.random(state)
                res = tuple(-1 if i == move else -s for i, s in enumerate(state))
                self.assertIsNotNone(move)
                state, outcome = an.step(state, move)
                count += 1
                if state is not None:
                    self.assertEqual(state, res)
                    self.assertIsNone(outcome)
                else:
                    self.assertIsNone(state)
                    self.assertIn(outcome, (-1, 0, 1))
            self.assertLessEqual(count, 9)

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
                ((0, 0, 1, 0, 0, 1, 0, 0, 0), 8),
                ((1, 0, 0, 0, 0, 0, 0, 0, 1), 4),
                ((1, 0, 0, 0, 1, 0, 0, 0, 0), 8)]
        for state, move in wins:
            self.assertEqual(an.step(state, move), (None, +1))

    def test_draw(self):
        an = AlphaNaughts()
        draw = [((1, -1, 1, 1, -1, -1, -1, 1, 0), 8),
                ((0, -1, 1, -1, -1, 1, -1, 1, -1), 0),
                ((-1, 0, 1, 1, -1, -1, -1, 1, 1), 1),
                ((-1, 1, 1, 1, -1, -1, -1, 0, 1), 7)]
        for state, move in draw:
            self.assertEqual(an.step(state, move), (None, 0))


if __name__ == '__main__':
    unittest.main()
