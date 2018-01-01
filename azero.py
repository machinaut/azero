#!/usr/bin/env python

from collections import defaultdict


num_simulations = 100
GAMES_PER_UPDATE = 100


class Edge:
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0


class Tree:
    def __init__(self, parent):
        assert isinstance(parent, Tree)
        self.parent = parent
        self.prior, self.value = None, None
        self.edges = defaultdict(Edge)  # XXX does this even work?

    def predict(self, model):
        ''' Use model to get prior probabilities and value '''
        self.prior, self.value = model(self.state)

    def isLeaf(self):
        ''' Is this a leaf node '''
        return self.prior is None

    def select(self):
        ''' Select move and return it and its next node '''
        # XXX TODO
        pass

    def simulate(self):
        ''' Simulate a game until leaf is reached, return leaf, trajectory '''
        current = self
        trajectory = []
        while not current.isLeaf():
            move, current = current.select()
            trajectory.append(move)
        return current, trajectory

    def search(self, model):
        ''' Run a whole MCTS and return probabilities for each move '''
        for _ in range(num_simulations):
            # TODO: run simulations in multiple threads in parallel
            # Paper says they use ~8 at once, and use virtual loss.
            leaf, trajectory = self.simulate()
            leaf.predict(model)


class Game:
    ''' Interface for a game used by alphazero '''
    def start(self):
        ''' Return a start state '''
        raise NotImplementedError()

    def valid(self, state):
        ''' Return a boolean array of action validity '''
        raise NotImplementedError()


class Model:
    ''' Interface for a model used by alphazero '''
    def __init__(self, game):
        ''' Initialize the model (uses game for number of actions, etc) '''
        raise NotImplementedError()

    def model(self, state):
        ''' Return action probability vector and outcome prediction '''
        raise NotImplementedError()

    def update(self, games):
        ''' Update model given games (lists of pairs of probs, outcome)'''
        raise NotImplementedError()


class AlphaZero:
    def __init__(self, game, model):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self.game = game
        self.model = model

    def play(self):
        ''' Self-play a game, return probabilities and outcome '''
        probs = []  # Pairs of (state, action probabilities from search)
        outcome = None  # +1 for Win, -1 for Loss
        return probs, outcome

    def train(self):
        games = [self.play() for _ in range(GAMES_PER_UPDATE)]
        self.model.update(games)


if __name__ == '__main__':
    print('hi')
