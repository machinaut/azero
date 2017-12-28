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

    def valid(self, state):
        ''' Return a binary mask for valid moves given a state '''
        # This is super easy because of numpy operators
        return np.asarray(state) == 0

    def step(self, state, move):
        ''' Get the next game state given a move '''
        # Invert all the pieces on the board (P1 <-> P2 swap)
        # Make the move -1 (because the previous player placed it)
        assert state[move] == 0, 'Invalid move {} state {}'.format(move, state)
        for a, b in self.three(move):
            if state[a] == 1 and state[b] == 1:
                return None, +1
        next_state = tuple(-1 if move == i else -s for i, s in enumerate(state))
        # Check for empty board slots
        for s in next_state:
            if s == 0:  # There is an empty slot
                return tuple(next_state), None
        return None, 0  # No free slots, game is a draw

    def three(self, move):
        ''' Return a generator of pairs to check for win detection '''
        y0, x0 = divmod(move, 3)
        y1 = (y0 + 1) % 3
        y2 = (y1 + 1) % 3
        x1 = (x0 + 1) % 3
        x2 = (x1 + 1) % 3
        yield (y0 * 3 + x1, y0 * 3 + x2)
        yield (y1 * 3 + x0, y2 * 3 + x0)
        if move in (0, 4, 8):
            yield (y1 * 3 + x1, y2 * 3 + x2)
        if move in (2, 4, 6):
            yield (y1 * 3 + x2, y2 * 3 + x1)

    def print(self, state):
        for i in range(3):
            print('%+1d %+1d %+1d' % state[i * 3:i * 3 + 3])
