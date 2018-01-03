#!/usr/bin/env python

import numpy as np
from azero import AlphaZero
from game import TicTacToe
from model import Memorize


def play(azero):
    ''' Play a game with the human '''
    while True:
        print('\nNew Game!')
        print('Doc:', azero.game.__doc__)
        human = np.random.random() < .5  # Human starts
        state = azero.game.start()
        while state is not None:
            print('Turn:', 'human' if human else 'azero')
            print('State:', azero.game.human(state))
            if human:
                action = None
                while action is None:
                    valid = game.valid(state)
                    print('Valid:', np.flatnonzero(np.asarray(valid)))
                    action = int(input('Move:'))
                    if not valid[action]:
                        print('Invalid move, try again.')
                        action = None
            else:
                probs, _ = azero.model.model(state)
                probs += np.where(azero.game.valid(state), 0, -np.inf)
                print('Probs:', probs)
                action = np.argmax(probs)
                print('Move:', action)
            human = not human  # Switch players
            state, outcome = game.step(state, action)
        player = 'human' if human ^ (outcome > 0) else 'azero'
        if outcome != 0:
            print('Outcome:', player, 'wins!')
        else:
            print('Outcome: draw!')


if __name__ == '__main__':
    game = TicTacToe()
    model = Memorize(game)
    azero = AlphaZero(game, model)
    azero.train()
    play(azero)
