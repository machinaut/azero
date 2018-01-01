#!/usr/bin/env python

import numpy as np
from azero import Model


class NearestNeighbor(Model):
    ''' Return move probs and model from nearest-neighbor search '''
    def __init__(self, game):
        self.data = {}  # Map from tuple(state) -> (probs, value)
        self.n_obs = len(game.start())
        self.n_act = len(game.valid(game.start()))

    def model(self, state):
        for tstate, result = 
        probs = []  # Array of predicted probabilities
        outcome = 0  # Scalar from -1 (loss) to +1 (win)
        return probs, outcome
