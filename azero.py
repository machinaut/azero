#!/usr/bin/env python

import numpy as np
from util import softmax, sample_probs


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
                 eps=1e-6,
                 sims_per_search=100):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self._game = game
        self._model = model
        self.c_puct = c_puct
        self.tau = tau
        self.eps = eps
        self.sims_per_search = sims_per_search

    @classmethod
    def make(cls, game_cls, model_cls, *args, **kwargs):
        ''' Convenience method to build from game and model classes '''
        game = game_cls()
        model = model_cls(game.n_action, game.n_view, game.n_player)
        return cls(game=game, model=model, *args, **kwargs)

    def model(self, state, player):
        ''' Wrap the model to give the proper view and mask actions '''
        valid = self._game.valid(state, player)
        view = self._game.view(state, player)
        logits, value = self._model.model(view)
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

    def play(self):
        '''
        Play a whole game, and get states on which to update
        Return tuple of:
            trajectory - (observation, probabilities) for each step
            outcome - final reward for each player
        '''
        trajectory = []
        state, player, outcome = self._game.start()
        while outcome is None:
            probs, _ = self.search(state, player)
            action = sample_probs(probs)
            obs = self._game.view(state, player)
            trajectory.append((obs, probs))
            state, player, outcome = self._game.step(state, player, action)
        return trajectory, outcome

    def play_multi(self, n=10):
        '''
        Play multiple whole games, return a list of game results.
        See play() for result of a single game.
        '''
        return [self.play() for _ in range(n)]
