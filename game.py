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
        return np.zeros(1)

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
        raise ValueError('Invalid state/action')
