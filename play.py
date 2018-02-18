#!/usr/bin/env python


def play(game):
    print('Playing:', game.__class__.__name__)
    print('Docstring:', game.__doc__)
    state, player, outcome = game.start()
    while outcome is None:
        print('State:', game.human(state))
        print('Player:', player)
        valid = game.valid(state, player)
        print('Valid:', valid)
        sparse = tuple(i for i in range(len(valid)) if valid[i])
        action = int(input('Action ' + str(sparse) + ':'))
        state, player, outcome = game.step(state, player, action)
    print('Outcome:', outcome)
