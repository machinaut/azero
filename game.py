#!/usr/bin/env python

import numpy as np


class Game:
    ''' Interface for a game used by alphazero '''
    def start(self):
        ''' Return a start state '''
        raise NotImplementedError()

    def valid(self, state):
        ''' Return a boolean array of action validity '''
        raise NotImplementedError()

    def step(self, state, action):
        ''' Return a pair of (next state or None), (outcome or None) '''
        raise NotImplementedError()

    def human(self, state):
        ''' Print out a human-readable state '''
        return str(state)


class Bandit(Game):
    '''
    Perfect-information slot machine:
    State: Action which wins (all other actions lose)
    Action: Which lever to pull
    '''
    def start(self):
        return np.random.choice(10, 1)

    def valid(self, state):
        return np.ones(10)

    def step(self, state, action):
        return None, +1 if state[0] == action else -1


class RockPaperScissors(Game):
    '''
    Turn-based Rock-Paper-Scissors (second player should always win)
    State: -1: First players turn, 0: rock, 1: paper, 2: scissors
    Actions: 0: rock, 1: paper, 2: scissors
    '''
    def start(self):
        return np.array([-1])

    def valid(self, state):
        return np.ones(3)

    def step(self, state, action):
        if state[0] < 0:
            return np.array([action]), None
        if state[0] == action:
            return None, 0  # Tie
        if state[0] == (action - 1) % 3:
            return None, 1  # Win
        if state[0] == (action + 1) % 3:
            return None, -1  # Loss

    def human(self, state):
        return {-1: 'Start', 0: 'Rock', 1: 'Paper', 2: 'Scissors'}[state[0]]


class TicTacToe(Game):
    '''
    Tic-Tac-Toe
    State: 10 vector of all 9 positions in order, then player number
    Actions: Board position to play in
    '''
    WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
            (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))

    def __init__(self):
        self.memo = {}

    def start(self):
        state = np.array([0] * 9 + [1])
        state.setflags(write=False)
        return state

    def valid(self, state):
        return state[:9] == 0

    def step(self, state, action):
        key = (tuple(state), action)
        if key in self.memo:
            return self.memo[key]
        assert state[action] == 0, 'Bad step {} {}'.format(state, action)
        player = state[9]
        state = state.copy()
        state[action] = player
        for a, b, c in self.WINS:
            if state[a] == state[b] == state[c] == player:
                result = None, +1
                break
        else:
            state[9] = -player  # Next players turn
            state.setflags(write=False)
            if 0 not in state:
                result = None, 0  # Draw, no more available moves
            else:
                result = state, None
        self.memo[key] = result
        return result

    def human(self, state):
        s = ''
        for i in range(0, 9, 3):
            s += '\n' + ' '.join(str(c) for c in state[i: i + 3])
        return s


def play(game):
    print('Playing:', type(game))
    print('Doc:', game.__doc__)
    state = game.start()
    while state is not None:
        print('State:', game.human(state))
        print('Valid:', np.flatnonzero(game.valid(state)))
        action = int(input('Move:'))
        state, outcome = game.step(state, action)
    print('Outcome:', outcome)


if __name__ == '__main__':
    play(TicTacToe())
