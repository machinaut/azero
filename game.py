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


class Count(Game):
    '''
    Count up to 9 from 0
    State: last number counted (starts at 0)
    Action: next number to count
    '''
    def start(self):
        return (0,)

    def valid(self, state):
        return (1,) * 10

    def step(self, state, action):
        if state[0] + 1 == action:
            if action == 9:
                return None, +1  # Win
            return (action,), None  # Next
        return None, -1  # Lose


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
    START = (0,) * 9 + (1,)
    WINS = ((0, 1, 2), (0, 3, 6), (0, 4, 8), (1, 4, 7),
            (2, 4, 6), (2, 5, 8), (3, 4, 5), (6, 7, 8))

    def __init__(self):
        self.memo = {}

    def start(self):
        return self.START

    def valid(self, state):
        return tuple(s == 0 for s in state[:9])

    def step(self, state, action):
        key = (state, action)
        result = self.memo.get(key, None)
        if result is None:
            assert state[action] == 0, 'Bad step {} {}'.format(state, action)
            player = state[9]
            board = tuple(player if i == action else s for i, s in enumerate(state[:9]))
            for a, b, c in self.WINS:
                if board[a] == board[b] == board[c] == player:
                    result = None, +1
                    break
            else:
                if 0 not in board:
                    result = None, 0  # Draw, no more available moves
                else:
                    result = board + (-player,), None
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
        print('Valid:', np.flatnonzero(np.asarray(game.valid(state))))
        action = int(input('Move:'))
        state, outcome = game.step(state, action)
    print('Outcome:', outcome)


if __name__ == '__main__':
    play(Count())
