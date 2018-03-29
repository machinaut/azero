#!/usr/bin/env python

import numpy as np
from util import softmax, sample_probs


class Tree:
    ''' Data structure used during simulated games '''
    def __init__(self, prior, c_puct):
        self.c_puct = c_puct
        self.T = 0  # Total visits
        self.N = np.zeros(len(prior), dtype=int)  # Visit count
        self.W = np.random.randn(len(prior)) * 1e-8
        self.Q = np.random.randn(len(prior)) * 1e-8
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
    def __init__(self, game, model, seed=None,
                 c_puct=1.0,
                 tau=1.0,
                 eps=1e-6,
                 sims_per_search=300):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self.rs = np.random.RandomState(seed)
        self._game = game
        self._model = model
        self.c_puct = c_puct
        self.tau = tau
        self.eps = eps
        self.sims_per_search = sims_per_search

    @classmethod
    def make(cls, game_cls, model_cls, seed=None, *args, **kwargs):
        ''' Convenience method to build from game and model classes '''
        game = game_cls(seed=seed)
        model = model_cls(game.n_action, game.n_view, game.n_player, seed=seed)
        return cls(game=game, model=model, seed=seed, *args, **kwargs)

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
        if sum(valid):
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
        return [(-1, 1), (1, -1)][player]  # XXX hardcode 2 player loss

    def search(self, state, player, sims_per_search=None):
        ''' MCTS to generate move probabilities for a state '''
        if sims_per_search is None:
            sims_per_search = self.sims_per_search
        prior, _ = self.model(state, player)
        tree = Tree(prior, self.c_puct)
        for i in range(sims_per_search):
            self.simulate(state, player, tree)
        pi = np.power(tree.N, 1 / self.tau)
        if np.sum(pi) < 1e-10:
            #import ipdb; ipdb.set_trace()
            probs = np.zeros(pi.shape)
            probs[0] = 1.0
            return probs, tree
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
            action = sample_probs(probs, rs=self.rs)
            obs = self._game.view(state, player)
            trajectory.append((obs, probs))
            state, player, outcome = self._game.step(state, player, action)
        return trajectory, outcome

    def play_multi(self, n_games=10):
        '''
        Play multiple whole games, return a list of game results.
        See play() for result of a single game.
        '''
        games = []
        for i in range(n_games):
            print('playing game', i)
            games.append(self.play())
        return games

    def train(self, n_epochs=10, n_games=10):
        '''
        Train the model for a number of epochs of multi-play
        '''
        for i in range(n_epochs):
            games = self.play_multi(n_games=n_games)
            loss = self._model.update(games)
            print('epoch', i, 'loss', loss)
            eval_rollouts = [self.eval_play() for _ in range(10)]
            print('eval score', np.mean(eval_rollouts))

    def rollout(self):
        ''' Rollout a game against self and return final state '''
        state, player, outcome = self._game.start()
        while outcome is None:
            probs, _ = self.search(state, player)
            action = sample_probs(probs, rs=self.rs)
            state, player, outcome = self._game.step(state, player, action)
        return state

    def print_rollout(self):
        ''' Print out final board state '''
        print(self._game.human(self.rollout()))

    def eval_play(self):
        ''' Rollout game vs random agent '''
        state, player, outcome = self._game.start()
        random_agent = np.random.choice(2)
        while outcome is None:
            if player == random_agent:
                valid = self._game.valid(state, player)
                probs = np.array(valid, dtype=float) / sum(valid)
            else:
                probs, _ = self.search(state, player)
            action = sample_probs(probs, rs=self.rs)
            state, player, outcome = self._game.step(state, player, action)
        return -outcome[random_agent]


if __name__ == '__main__':
    from game import Checkers  # noqa
    from model import MLP, Uniform  # noqa
    azero = AlphaZero.make(Checkers, MLP)
    azero.train(n_epochs=10000, n_games=10)
    #azero.sims_per_search = 1000
    #total_outcomes = np.zeros(2)
    #for i in range(1, 100):
    #    _, outcome = azero.play()
    #    total_outcomes += outcome
    #    print('score', total_outcomes / i)
