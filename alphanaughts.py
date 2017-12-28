#!/usr/bin/env python

import numpy as np


'''
Notes for alphazero reimplementation.

Rough draft: game environment, data gathering, and model training.

Game environment:
- ignoring symmetries and convolutions
- observation:
    - board state encoded as vectors (lol)
- action:
    - discretized from all possible actions
    - mask out invalid actions from vector and renormalize

Search:
- outputs probability distribution over moves
- try to capture the empirical result of the sim?

Data gathering:
- Single model running on both sides of the board.
- Move selection based on:
    - High move probability according to policy
    - High value (averaged over leaf nodes from this state)
    - Low visit count
- Whole game rollouts (not subgames, for simplicity)

Model training:
Optimize loss = MSE(outcome) + cross_entropy(actions) + L2_reg
'''

# Hyperparameters and configuration parameters
DEFAULT_CONFIG = {
    'num_leafs': 10,  # Number of roll-out games per search
    'num_games': 10,  # Number of whole games to play per epoch
    'num_epoch': 10,  # Number of epochs to play during training run
}


class AlphaNaughts:
    '''
    AlphaZero Naughts and Crosses (tic-tac-toe).

    Game state definition:
        Numbered board positions in a vector:
            0 1 2
            3 4 5
            6 7 8
        Positions are encoded as:
            Empty: 0, P1 (my player): 1, P2 (opponent): -1
            (to switch sides, negate the vector!)
        Example:
            O - X
            O X O
            - - X
            (Because X goes first we know it's X's turn)
            Encoded as [-1, 0, 1, -1, 1, -1, 0, 0, 1]
    '''
    START = (0,) * 9  # Empty board

    def __init__(self, config=None):
        if config is None:
            config = DEFAULT_CONFIG.copy()
        self.config = config
        # Indexed by state (as tuple of integers) -> (probs, count, value)
        #   probs - probability of actions in that state (vector)
        #   count - number of times this state has been visited
        #   value - estimated value of this state
        self._policy_data = {}

    def config(self, name):
        return self._config.get(name)

    def policy(self, state):
        ''' Function: state -> (action_probs, state_value) '''
        # YOLO XXX not even nearest neigbor.
        return self._policy_data.get(state, (np.ones(9) / 9, 0))

    def update(self, experience):
        ''' Update policy with experience '''
        for exp in experience:
            probs, value

    def train(self):
        for _ in range(self.config['num_epoch']):
            experience = []
            for _ in range(self.config['num_games']):
                exp = self.game()
                experience += exp
            self.update(experience)

    def game(self):
        ''' Play a whole game (with search every step) and return experience '''
        state = self.START  # Start with empty board
        experience = []  # list of state, probs, value
        while state is not None:  # set state to None when we're done
            emp_probs, emp_value, move = self.search(state)
            experience.append(state, emp_probs, emp_value)
            state = self.step(state, move)
        return experience

    def search(self, state):
        ''' Search from a given state, return probs, value and move selected '''
        # Initial policy selection = probability * value (of post-move state)
        probs, _ = np.asarray(self.policy(state))
        values = [self.policy(self.step(state, i))[1] for i in range(9)]
        counts = np.ones(9)
        result = np.zeros(9)
        select = probs * values
        for _ in range(self.config['num_leafs']):
            # Selection criterion = (prob * value + total_returns) / (1 + count)
            # Which should always be in [-1, +1]
            # This seems like a bad/biased measure, TODO get a better one
            move = self.sample(state, select)
            counts[move] += 1
            result = self.rollout(state, move)
            select = (probs * values + result) / counts
        return select,

    def rollout(self, state, move):
        ''' Rollout a fast (no search) game and return result '''
        state = self.step(state, move)
        count = 0
        while state is not None:
            move = self.sample(state, self.policy(state)[0])
            count += 1
        # TODO XXX maybe have step return (next, outcome)?
        # TODO: unittest this end detection and rollout code
        return self.won(state) * (-1 ** (count % 2))

    def sample(self, state, probs):
        ''' Sample a valid move from a vector of probabilities '''
        # TODO: probs -> logits and softmax it
        masked = np.asarray(probs) * self.valid(state)
        norm = masked / np.sum(masked)
        return np.random.choice(len(norm), p=norm)

    def random(self, state):
        ''' Sample a random valid move '''
        if self.done(state) is not None:
            return None
        valid = self.valid(state)
        return np.random.choice(len(valid), p=(valid / np.sum(valid)))

    def valid(self, state):
        ''' Return a binary mask for valid moves given a state '''
        # This is super easy because of numpy operators
        return np.asarray(state) == 0

    def step(self, state, move):
        '''
        Get the next game state given a move.
        Returns tuple of (next_state, outcome).
        If the game is done:
            next_state is None
            outcome is +1 for last move won, -1 for last move lost, 0 for draw
        If the game is not done:
            next_state is the next board state with the players swapped
            outcome is None
        '''
        assert state[move] == 0, 'Invalid move {} state {}'.format(move, state)
        next_state = tuple(1 if i == move else s for i, s in enumerate(state))
        outcome = self.done(next_state)
        if outcome is not None:
            return None, outcome
        return tuple(-s for s in next_state), None

    def done(self, state):
        '''
        Return if game is done at a state:
            +1 - Player 1 won
            -1 - Player 2 won
             0 - Game ended in a draw
            None - game is not done yet
        '''
        wins = ((0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal
                (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical
                (0, 4, 8), (2, 4, 6))  # Diagonal
        for player in (+1, -1):
            for a, b, c in wins:
                if state[a] == state[b] == state[c] == player:
                    return player
        if state.count(0) == 0:
            return 0  # Draw, no more available moves
        return None

    def print(self, state):
        for i in range(3):
            print('%+1d %+1d %+1d' % state[i * 3:i * 3 + 3])
