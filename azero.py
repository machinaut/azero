#!/usr/bin/env python

import numpy as np
from util import view2obs, softmax


class Tree:
    ''' Data structure used during simulated games '''
    def __init__(self, prior, c_puct):
        self.c_puct = c_puct
        self.T = 0  # Total visits
        self.N = np.zeros(len(prior), dtype=int)  # Visit count
        self.W = np.zeros(len(prior))  # Total action-value
        self.Q = np.zeros(len(prior))  # Mean action-value == W / N
        self.P = np.array(prior)  # Scaled prior == prior / (1 + N)
        self.prior = np.array(prior)
        self.children = dict()

    def leaf(self, action, prior):
        self.children[action] = Tree(prior, c_puct=self.c_puct)

    @property
    def U(self):  # Upper Confidence Bound
        return self.c_puct * np.sqrt(self.T) * self.P

    @property
    def values(self):  # Mean action value + UCB == Q + U
        return self.Q + self.U

    def select(self, valid):
        ''' Select given valid moves and return action, child '''
        action = np.argmax(np.where(valid, self.values, -np.inf))
        return action, self.children.get(action, None)

    def backup(self, action, value):
        ''' Backup results of a simulation game '''
        self.T += 1
        self.N[action] = n = self.N[action] + 1
        self.W[action] = w = self.W[action] + value
        self.Q[action] = w / n
        self.P[action] = self.prior[action] / (1 + n)


class AlphaZero:
    def __init__(self, game, model,
                 c_puct=1.0,
                 tau=1.0,
                 sims_per_search=100):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self._game = game
        self._model = model
        self.c_puct = c_puct
        self.tau = tau
        self.sims_per_search = sims_per_search

    def model(self, state, player):
        ''' Wrap the model to give the proper view and mask actions '''
        valid = self._game.valid(state, player)
        view = self._game.view(state, player)
        obs = view2obs(view, player)
        logits, value = self._model.model(obs)
        probs = softmax(logits, valid)
        return probs, value

    def simulate(self, state, player, tree):
        '''
        Simulate a game by traversing tree
            state - game state tuple
            player - current player index
            tree - MCTS tree rooted at current state
        returns
            values - player-length list of values
        '''
        valid = self._game.valid(state, player)
        action, child = tree.select(valid)
        if child is None:
            prior, values = self.model(state, player)
            tree.leaf(action, prior)
        else:
            state, next_player, values = self._game.step(state, player, action)
            if values is None:
                values = self.simulate(state, next_player, child)
        tree.backup(action, values[player])
        return values

    def search(self, state, player):
        ''' MCTS to generate move probabilities for a state '''
        prior, _ = self.model(state, player)
        tree = Tree(prior, self.c_puct)
        for _ in range(self.sims_per_search):
            self.simulate(state, player, tree)
        pi = np.power(tree.N, 1 / self.tau)
        probs = pi / np.sum(pi)
        return probs, tree
