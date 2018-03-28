#!/usr/bin/env python

import random
import argparse
from game import games
from model import Uniform
from azero import AlphaZero
import numpy as np


def play(game):
    model = Uniform.make(game)
    azero = AlphaZero(game, model)
    print(azero)
    print('Playing:\n', game.__class__.__name__)
    print('Docstring:\n', game.__doc__)
    state, player, outcome = game.start()
    while outcome is None:
        print('Human:\n' + game.human(state, player))
        # print('State:\n' + str(state) + '\n')
        # print('View:\n', game.view(state, player))
        print('Player:\n', player)
        valid = game.valid(state, player)
        # print('Valid:\n', valid)
        probs, _ = azero.search(state, player, sims_per_search=1000)
        print('Azero:', np.argmax(probs))
        sparse = tuple(i for i in range(len(valid)) if valid[i])
        action = None
        while action not in sparse:
            try:
                action = int(input('Action ' + str(sparse) + ': '))
            except ValueError:
                pass
        state, player, outcome = game.step(state, player, action)
    if state is not None:
        print('Final State:\n' + game.human(state, player))
    print('Outcome:')
    for i, val in enumerate(outcome):
        print('  Player', i, ':', val)


def main():
    names = [g.__name__ for g in games]
    choices = {g.__name__: g for g in games}
    default = random.choice(names)
    parser = argparse.ArgumentParser()
    parser.add_argument('game', nargs='?', default=default, choices=names,
                        help='Which game to play (default is randomly chosen)')
    args = parser.parse_args()
    game = choices[args.game]()
    play(game)


if __name__ == '__main__':
    main()
