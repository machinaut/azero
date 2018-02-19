#!/usr/bin/env python

import numpy as np
from util import view2obs, softmax


SIMS_PER_SEARCH = 10
C_PUCT = 1.0  # PUCT coefficient controls exploration in search
TAU = 1.0  # Temperature, controls exploration in move selection


class Tree:
    ''' Data structure used during simulated games '''
    def __init__(self, prior):
        self.T = 0  # Total visits
        self.N = np.zeros(len(prior), dtype=int)  # Visit count
        self.W = np.zeros(len(prior))  # Total action-value
        self.Q = np.zeros(len(prior))  # Mean action-value == W / N
        self.P = np.array(prior)  # Scaled prior == prior / (1 + N)
        self.prior = np.array(prior)
        self.children = dict()

    def leaf(self, action, prior):
        self.children[action] = Tree(prior)

    @property
    def U(self):
        return C_PUCT * np.sqrt(self.T) * self.P

    def select(self):
        ''' Select given valid moves and return action, child '''
        action = np.argmax(self.Q + self.U)
        return action, self.children.get(action, None)

    def backup(self, action, value):
        ''' Backup results of a simulation game '''
        self.T += 1
        self.N[action] = n = self.N[action] + 1
        self.W[action] = w = self.W[action] + value
        self.Q[action] = w / n
        self.P[action] = self.prior[action] / (1 + n)


class AlphaZero:
    def __init__(self, game, model):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self._game = game
        self._model = model

    def step(self, state, player, action):
        ''' Wrap game action to check for valid moves '''
        if not self._game.valid(state, player)[action]:
            return state, player, -1  # Kinda hack but it works
        return self._game.step(state, player, action)

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
            value - value of leaf state
            player - player the value applies to
            outcome - player index of the winner or None
        '''
        action, child = tree.select()
        if child is None:
            prior, value = self.model(state, player)
            tree.leaf(action, prior)
            return value, player, None
        state, next_player, outcome = self.step(state, player, action)
        if outcome is not None:
            value = 1 if outcome == player else -1
            tree.backup(action, value)
            return value, player, outcome
        value, v_player, outcome = self.simulate(state, next_player, child)
        if v_player == player:
            tree.backup(action, value)
        return value, v_player, outcome

    def search(self, state, player):
        ''' MCTS to generate move probabilities for a state '''
        prior, _ = self.model(state, player)
        tree = Tree(prior)
        for _ in range(SIMS_PER_SEARCH):
            self.simulate(state, player, tree)
        pi = np.power(tree.N, 1 / TAU)
        probs = pi / np.sum(pi)
        return probs

    def sample(self, state, player):
        ''' Return a sampled action from a search '''
        probs = self.search(state, player)
        return np.random.choice(range(len(probs)), p=probs)
