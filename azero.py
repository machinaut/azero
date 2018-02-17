#!/usr/bin/env python

import numpy as np
from math import sqrt
from collections import defaultdict
from util import select


NUM_UPDATES = 10
GAMES_PER_UPDATE = 10
GAMES_PER_EVAL = 10
SIMS_PER_SEARCH = 100
C_PUCT = 1.0  # PUCT coefficient controls exploration in search
TAU = 1.0  # Temperature, controls exploration in move selection


class Tree:
    ''' Data structure used during simulated games '''
    def isLeaf(self):
        return not hasattr(self, 'children')

    def expand(self, state, probs, value, valid):
        ''' Expand tree with results of model '''
        self.children = defaultdict(Tree)  # Map from action -> sub-Tree()
        self.state = state
        self.prior = np.asarray(probs) * valid
        self.value = value
        self.valid = valid
        self.T = 0  # Total of all N(s, a) of children
        self.N = np.zeros(len(probs), dtype=int)
        self.W = np.zeros(len(probs))
        self.Q = np.zeros(len(probs))
        self.P = np.array(self.prior)

    def select(self):
        ''' Select given valid moves and return action, child '''
        U = C_PUCT * sqrt(self.T) * self.P
        action = np.argmax(np.where(self.valid, self.Q + U, -np.inf))
        assert self.valid[action], 'Bad {} {}'.format(self.valid, action)
        return action, self.children[action]

    def debug(self):
        ''' Output the upper confidence bound values for debugging '''
        U = C_PUCT * sqrt(self.T) * self.P
        values = np.where(self.valid, self.Q + U, -np.inf)
        return values

    def backup(self, action, value):
        ''' Backup results of a simulation game '''
        self.T += 1
        self.N[action] = N = self.N[action] + 1
        self.W[action] = W = self.W[action] + value
        self.Q[action] = W / N
        self.P[action] = self.prior[action] / (1 + N)


class AlphaZero:
    def __init__(self, game, model):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self.game = game
        self.model = model

    def simulate(self, state, player, tree):
        ''' Simulate a game by traversing tree '''
        assert isinstance(tree, Tree)
        if tree.isLeaf():
            probs, value = self.model.model(state)
            valid = self.game.valid(state)
            tree.expand(state, probs, value, valid)
            return value
        action, child = tree.select()
        next_state, next_player, outcome = self.game.step(state, action)
        if next_state is None:
            value = outcome
        else:
            # player * next_player is 1 if they are the same (1,1) or (-1,-1)
            # and is -1 if they are different (-1,1) or (1,-1)
            value = self.simulate(next_state, next_player, child) * player * next_player
        tree.backup(action, value)
        return value

    def search(self, state, player):
        ''' MCTS to generate move probabilities for a state '''
        tree = Tree()
        for _ in range(SIMS_PER_SEARCH):
            self.simulate(state, player, tree)
        pi = np.power(tree.N, 1 / TAU)
        probs = pi / np.sum(pi)
        return probs

    def sample(self, state, probs):
        ''' Sample a valid action from a vector of action probabilities '''
        move = np.random.choice(len(probs), p=probs / np.sum(probs))
        assert self.game.valid(state)[move], 'bad {} {}'.format(probs, state)
        return move

    def play(self):
        ''' Self-play a game, return probabilities and outcome '''
        trajectory = []  # List of pairs of (state, probabilities from search)
        state = self.game.start()
        player = 1
        while state is not None:
            probs = self.search(state, player)
            trajectory.append((state, player, probs))
            action = self.sample(state, probs)
            state, player, outcome = self.game.step(state, action)
        return trajectory, outcome

    def eval(self):
        ''' Return win rate against random agent '''
        total = 0
        for i in range(GAMES_PER_EVAL):
            state = self.game.start()
            player = 1
            playing = bool(i % 2)
            while state is not None:
                if playing:
                    probs = self.search(state, player)
                    action = self.sample(state, probs)
                else:
                    action = select(self.game.valid(state))
                old_player = player
                state, player, outcome = self.game.step(state, action)
                if old_player != player:
                    playing = not playing
            total += -outcome if playing else outcome
        return (total + GAMES_PER_EVAL) / (2 * GAMES_PER_EVAL)

    def train(self):
        for i in range(NUM_UPDATES):
            print('Update:', i, '/', NUM_UPDATES)
            print('Eval', self.eval())
            games = []
            for j in range(GAMES_PER_UPDATE):
                games.append(self.play())
            self.model.update(games)


if __name__ == '__main__':
    from game import TicTacToe  # noqa
    from model import Memorize  # noqa
    NUM_UPDATES = 2  # Faster for profiling purposes
    game = TicTacToe()
    model = Memorize(game)
    az = AlphaZero(game, model)
    az.train()
