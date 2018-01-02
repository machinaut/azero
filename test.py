#!/usr/bin/env python

from azero import AlphaZero
from game import RockPaperScissors
from model import NearestNeighbor


if __name__ == '__main__':
    game = RockPaperScissors()
    model = NearestNeighbor(game)
    azero = AlphaZero(game, model)
    azero.train()
