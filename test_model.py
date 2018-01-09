#!/usr/bin/env python

import unittest
from model import models
from game import games


class TestModel(unittest.TestCase):
    def test_shape(self):
        ''' Test that the model gives vectors of correct length '''
        for game_class in games:
            for model_class in models:
                game = game_class()
                model = model_class(game)
                start = game.start()
                valid = game.valid(start)
                probs, _ = model.model(start)
                self.assertEqual(len(valid), len(probs))


if __name__ == '__main__':
    unittest.main()
