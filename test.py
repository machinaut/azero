#!/usr/bin/env python

import numpy as np
from azero import AlphaZero
from game import RockPaperScissors
from model import NearestNeighbor


def play(azero):
    ''' Play a game with the human '''
    while True:
        print('\nNew Game!')
        if np.random.random() < .5:  # Human starts
            player = 'human'
        else:
            player = 'azero'
        print('Player', player, 'goes first!')
        state = azero.game.start()
        while state is not None:
            print('State:', state)
            if player == 'human':
                print('Valid:', np.flatnonzero(game.valid(state)))
                action = int(input('Move:'))
                player = 'azero'
            elif player == 'azero':
                probs, _ = azero.model.model(state)
                print('Probs:', probs)
                action = np.argmax(probs)
                print('Move:', action)
                player = 'human'
            else:
                raise ValueError('invalid player!')
            state, outcome = game.step(state, action)
        if outcome > 0:
            print('Outcome:', player, 'loses!')
        elif outcome < 0:
            print('Outcome:', player, 'wins!')
        else:
            print('Outcome: draw!')


if __name__ == '__main__':
    game = RockPaperScissors()
    model = NearestNeighbor(game)
    azero = AlphaZero(game, model)
    azero.train()
    play(azero)
