#!/usr/bin/env python

import numpy as np
from azero import Game


class TicTacToe(Game):
    '''
    Game state definition:
        Numbered board positions in a vector:
            0 1 2
            3 4 5
            6 7 8
              9    - last vector is "current player" (whose turn it is)
        Positions are encoded as:
             0 : empty
            +1 : first player
            -1 : second player
    '''
    START = np.array([0] * 9 + [1])  # Empty board, player 1's turn
    START.setflags(write=False)  # Make it write-only

    def start(self):
        return self.START

    def valid(self, state):
        return state[:9] == 0  # Any empty space is a valid move


if __name__ == '__main__':
    print('hi')
