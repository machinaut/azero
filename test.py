#!/usr/bin/env python

import numpy as np
from azero import AlphaZero
from game import TicTacToe
from model import Memorize


def play(azero):
    ''' Play a game with the human '''
    while True:
        print('\nNew Game!')
        if np.random.random() < .5:  # Human starts
            player = 'human'
        else:
            player = 'azero'
        state = azero.game.start()
        while state is not None:
            print('Turn:', player)
            print('State:', azero.game.human(state))
            if player == 'human':
                print('Valid:', np.flatnonzero(game.valid(state)))
                action = int(input('Move:'))
                player = 'azero'
            elif player == 'azero':
                probs, _ = azero.model.model(state)
                probs += np.where(azero.game.valid(state), 0, -np.inf)
                print('Probs:', probs)
                action = np.argmax(probs)
                print('Move:', action)
                player = 'human'
            else:
                raise ValueError('invalid player!')
            state, outcome = game.step(state, action)
        if outcome < 0:
            print('Outcome:', player, 'wins!')
        elif outcome > 0:
            print('Outcome:', player, 'loses!')
        else:
            print('Outcome: draw!')


if __name__ == '__main__':
    game = TicTacToe()
    model = Memorize(game)
    azero = AlphaZero(game, model)
    azero.train()
    play(azero)
