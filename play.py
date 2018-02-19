#!/usr/bin/env python

import random
import argparse
from game import games


def play(game):
    print('Playing:', game.__class__.__name__)
    print('Docstring:', game.__doc__)
    state, player, outcome = game.start()
    while outcome is None:
        print('State:', game.human(state))
        print('View:', game.view(state, player))
        print('Player:', player)
        valid = game.valid(state, player)
        print('Valid:', valid)
        sparse = tuple(i for i in range(len(valid)) if valid[i])
        action = None
        while action not in sparse:
            try:
                action = int(input('Action ' + str(sparse) + ': '))
            except ValueError:
                pass
        state, player, outcome = game.step(state, player, action)
    print('Outcome:', outcome)


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
