#!/usr/bin/env python

GAMES_PER_UPDATE = 100


class AlphaZero:
    def __init__(self, game, model):
        ''' Train a model to play a game with the AlphaZero algorithm '''
        self.game = game
        self.model = model

    def play(self):
        ''' Self-play a game, return probabilities and outcome '''
        trajectory = []  # List of pairs of (state, probabilities from search)
        outcome = None  # +1 for Win, -1 for Loss
        return trajectory, outcome

    def train(self):
        games = [self.play() for _ in range(GAMES_PER_UPDATE)]
        self.model.update(games)


if __name__ == '__main__':
    print('hi')
