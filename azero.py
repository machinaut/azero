#!/usr/bin/env python

import numpy as np
from collections import defaultdict


NUM_UPDATES = 100
GAMES_PER_UPDATE = 10
GAMES_PER_EVAL = 10
SIMS_PER_SEARCH = 10
C_PUCT = 1.5  # PUCT coefficient controls exploration in search
TAU = 1.0  # Temperature, controls exploration in move selection


class Tree:
    ''' Data structure used during search step '''
    def isLeaf(self):
        return not hasattr(self, 'children')

    def expand(self, probs, value):
        ''' Expand tree with results of model '''
        self.children = defaultdict(Tree)  # Map from action -> sub-Tree()
        self.prior = probs
        self.value = value
        self.T = 0  # Total of all N(s, a) of children
        self.N = np.zeros(len(probs))
        self.W = np.zeros(len(probs))
        self.Q = np.zeros(len(probs))

    def select(self, valid):
        ''' Select given valid moves and return action, child '''
        U = C_PUCT * np.sqrt(self.T) * self.prior / (1 + self.N)
        Q = np.where(self.N > 0, self.W / self.N, 0)
        action = np.argmax(Q + U + np.where(valid, 0, -np.inf))
        return action, self.children[action]

    def backup(self, action, value):
        ''' Backup results of a simulation game '''
        self.T += 1
        self.N[action] += 1
        self.W[action] += value
        self.Q[action] = self.W[action] / self.N[action]

    def probs(self):
        ''' Return move probabilities '''
        pi = np.power(self.N, 1 / TAU)
        return pi / np.sum(pi)


class AlphaZero:
    def __init__(self, game, model):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self.game = game
        self.model = model

    def simulate(self, tree, state):
        ''' Simulate a game by traversing tree '''
        if tree.isLeaf():
            probs, value = self.model.model(state)
            tree.expand(probs, value)
            return value
        action, child = tree.select(self.game.valid(state))
        next_state, outcome = self.game.step(state, action)
        if next_state is None:
            value = outcome
        else:
            value = -self.simulate(child, next_state)
        tree.backup(action, value)
        return value

    def search(self, state, tree):
        ''' MCTS to generate move probabilities for a state '''
        for _ in range(SIMS_PER_SEARCH):
            self.simulate(tree, state)
        return tree.probs(), tree

    def sample(self, state, probs):
        ''' Sample a valid action from a vector of action probabilities '''
        masked = probs * self.game.valid(state)
        return np.random.choice(len(masked), p=masked / np.sum(masked))

    def play(self):
        ''' Self-play a game, return probabilities and outcome '''
        trajectory = []  # List of pairs of (state, probabilities from search)
        state = self.game.start()
        tree = Tree()
        while state is not None:
            probs, tree = self.search(state, tree)
            trajectory.append((state, probs))
            action = self.sample(state, probs)
            state, outcome = self.game.step(state, action)
            tree = tree.children[tree]  # Re-use subtree for chosen action
        return trajectory, outcome

    def eval(self):
        total = 0
        for i in range(GAMES_PER_EVAL):
            state = self.game.start()
            playing = bool(i % 2)
            while state is not None:
                if playing:
                    probs, _ = self.search(state, Tree())
                    action = self.sample(state, probs)
                else:
                    valid = self.game.valid(state)
                    action = np.random.choice(len(valid), p=valid / np.sum(valid))
                state, outcome = self.game.step(state, action)
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
