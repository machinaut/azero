#!/usr/bin/env python

import random
import numpy as np
from itertools import zip_longest, product


class Game:
    ''' Interface class for a game to be optimized by alphazero algorithm '''
    n_action = 0  # Number of possible actions
    n_state = 0  # Size of the state tuple
    n_player = 1  # Number of players
    n_view = 0  # Size of the observation of each player

    def __init__(self, seed=None) -> None:
        self.random = random.Random(seed)

    def check(self, state, player, outcome=None):
        '''
        Check if a combination of state and player and outcome are valid.
        Raises AssertionError() if not!
        '''
        if outcome is None:
            assert len(state) == self.n_state
            assert 0 <= player < self.n_player
            self._check(state, player)
        else:
            assert state is None
            assert player is None
            assert len(outcome) == self.n_player

    def _check(self, state, player):
        pass  # Optional: Implement in subclass

    def start(self):
        '''
        Start a new game, returns
            state - game state tuple
            player - index of the next player
            outcome - index of winning player or None if game is not over
        '''
        state, player, outcome = self._start()
        self.check(state, player, outcome)
        return state, player, outcome

    def _start(self):
        raise NotImplementedError('Implement in subclass')

    def step(self, state, player, action):
        '''
        Advance the game by one turn
            state - game state object (can be None)
            player - player making current move
            action - move to be played next
        Returns
            state - next state object (can be None)
            player - next player index or None if game is over
            outcome - index of winning player or None if game is not over
        '''
        self.check(state, player)
        assert 0 <= action < self.n_action
        state, player, outcome = self._step(state, player, action)
        self.check(state, player, outcome)
        return state, player, outcome

    def _step(self, state, player, action):
        raise NotImplementedError('Implement in subclass')

    def valid(self, state, player):
        '''
        Get a mask of valid actions for a given state.
            state - game state object
            player - next player index
        Returns:
            mask - tuple of booleans marking actions as valid or not
        '''
        self.check(state, player)
        valid = self._valid(state, player)
        assert len(valid) == self.n_action
        return valid

    def _valid(self, state, player):
        raise NotImplementedError('Implement in subclass')

    def view(self, state, player):
        '''
        Get a subset of the state that is observable to the player
            state - game state
            player - next player index
        Returns:
            view - subset of state visible to player
        '''
        self.check(state, player)
        view = self._view(state, player)
        assert len(view) == self.n_view
        return view

    def _view(self, state, player):
        return state  # Optional: Implement in subclass (default to full state)

    def human(self, state):
        ''' Print out a human-readable state '''
        return str(state)


class Null(Game):
    ''' Null game, always lose '''
    def _start(self):
        return None, None, (0,)


class Binary(Game):
    ''' Single move game, 0 - loses, 1 - wins '''
    n_action = 2

    def _start(self):
        return (), 0, None

    def _step(self, state, player, action):
        score = 1 if action == 1 else -1
        return None, None, (score,)

    def _valid(self, state, player):
        return (True, True)


class Flip(Game):
    ''' Guess a coin flip '''
    n_action = 2
    n_state = 1

    def _start(self):
        coin = self.random.randrange(2)
        return (coin,), 0, None

    def _step(self, state, player, action):
        score = 1 if action == state[0] else -1
        return None, None, (score,)

    def _valid(self, state, player):
        return (True, True)

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        assert 0 <= state[0] < 2


class Count(Game):
    ''' Count to 3 '''
    n_action = 3
    n_state = 1
    n_view = 1

    def _start(self):
        return (0,), 0, None

    def _step(self, state, player, action):
        count, = state
        if action != count:
            return None, None, (-1,)  # A loser is you
        if action == count == 2:
            return None, None, (1,)  # A winner is you
        return (count + 1,), 0, None

    def _valid(self, state, player):
        return (True, True, True)

    def _check(self, state, player):
        assert 0 <= state[0] < 3


class Narrow(Game):
    ''' Fewer choices every step '''
    n_action = 3
    n_state = 1
    n_view = 1

    def _start(self):
        return (3,), 0, None

    def _step(self, state, player, action):
        assert 0 <= action < state[0]
        if action == 0:
            return None, None, (-1,)
        return (action,), 0, None

    def _valid(self, state, player):
        return tuple(i < state[0] for i in range(3))

    def _check(self, state, player):
        assert 0 <= state[0] <= 3


class Matching(Game):
    ''' Matching Pennies '''
    n_action = 2
    n_state = 2
    n_player = 2

    def _start(self):
        return (0, 0), 0, None

    def _step(self, state, player, action):
        coin, _ = state
        if player == 0:
            return (action, 1), 1, None
        outcome = tuple(1 if action ^ coin ^ i else -1 for i in range(2))
        return None, None, outcome

    def _valid(self, state, player):
        return (True, True)

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        coin, current = state
        assert 0 <= coin < 2
        assert current == player


class Roshambo(Game):
    ''' Rock Paper Scissors '''
    n_action = 3
    n_state = 2
    n_player = 2

    def _start(self):
        return (0, 0), 0, None

    def _step(self, state, player, action):
        roshambo, _ = state
        if player == 0:
            return (action, 1), 1, None
        p0 = 1 if (action - 1) % 3 == roshambo else -1
        p1 = 1 if (action + 1) % 3 == roshambo else -1
        return None, None, (p0, p1)

    def _valid(self, state, player):
        return (True, True, True)

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        roshambo, current = state
        assert 0 <= roshambo < 3
        assert current == player


class Modulo(Game):
    ''' player mod 3 '''
    n_action = 3
    n_state = 2
    n_player = 3

    def _start(self):
        return (0, 0), 0, None

    def _step(self, state, player, action):
        total, _ = state
        total += action
        if player < 2:
            return (total, player + 1), player + 1, None
        return None, None, tuple(1 if total % 3 == i else -1 for i in range(3))

    def _valid(self, state, player):
        return (True, True, True)

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        total, current = state
        assert 0 <= total < 6
        assert current == player


class Connect3(Game):
    ''' Connect 4, but on a 5x4 grid.
        State is a 5x4 numpy array, with -1 as empty, 0 as player 0, and 1 as player 1.'''
    n_action = 5
    n_state = 20
    n_player = 2
    poss = [([0, 0, 0],[0, -1, -2]),
            ([0, -1, -2],[0, -1, -2]),
            ([1, 0, -1],[1, 0, -1]),
            ([2, 1, 0],[2, 1, 0]),
            ([0, 1, 2],[0, -1, -2]),
            ([-1, 0, 1],[1, 0, -1]),
            ([-2, -1, 0],[2, 1, 0]),
            ([-2, -1, 0],[0, 0, 0,]),
            ([-1, 0, 1],[0, 0, 0,]),
            ([0, 1, 2],[0, 0, 0,])]

    def _start(self):
        return np.ones((5,4),dtype=np.int8)*-1, 0, None

    def _step(self, state, player, action):
        assert state[action,-1] == -1
        new_piece = np.where(state[action]==-1)[0][0]
        state[action,new_piece] = player
        #Check for victory
        #Because I don't immediately see a simple way to check the whole board,
        #I'm going to just check the ten possible wins that involve the new piece.
        for poss in win_poss:
            if self._win(state, player, action, new_piece, poss[0], poss[1]):
                return state, None, player
        #Check for tie
        if not np.any(state[:,-1]==-1):
            return state, None, -1
        #Game continues
        return state, 1-player, None
    
    def _win(self, state, player, action, new_piece, x_set, y_set):
        win = True
        for piece in range(3):
            if not 0 <= action+x_set < state.shape[0]:
                return False
            if not 0 <= new_piece+y_set < state.shape[1]:
                return False
            if state[action+x_set,new_piece+y_set]:
                return False
        return True

    def _valid(self, state, player):
        return state[:,-1]==-1

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        pass

    def human(self, state):
        buffer = [' '.join('%+2d' % s for s in row) for row in state.transpose()]
        buffer.reverse()
        return '\n'.join(buffer)
        

class MNOP(Game):
    ''' Generalized tic-tac-toe '''
    def __init__(self, m=3, n=3, o=3, p=2, seed=None):
        super().__init__(seed=seed)
        assert m >= o and n >= o  # Otherwise game is unwinnable
        self.m = m  # board width
        self.n = n  # board height
        self.o = o  # goal
        self.n_player = self.p = p
        self.n_action = self.n_state = self.n_view = m * n

    def _start(self):
        return (-1,) * self.n_state, 0, None

    def _step(self, state, player, action):
        assert state[action] == -1
        m, n, o, p = self.m, self.n, self.o, self.p
        outcome = tuple(p - 1 if i == player else -1 for i in range(p))
        state = state[:action] + (player,) + state[action + 1:]
        s = tuple(zip_longest(*([iter(state)] * m)))
        for i, j in product(range(m - o + 1), range(n)):
            if all(s[i + k][j] == player for k in range(o)):
                return None, None, outcome
        for i, j in product(range(m), range(n - o + 1)):
            if all(s[i][j + k] == player for k in range(o)):
                return None, None, outcome
        for i, j in product(range(m - o + 1), range(n - o + 1)):
            if all(s[i + k][j + k] == player for k in range(o)):
                return None, None, outcome
            if all(s[i + k][j + o - k - 1] == player for k in range(o)):
                return None, None, outcome
        if state.count(-1) == 0:
            return None, None, (0,) * p
        return state, (player + 1) % p, None

    def _valid(self, state, player):
        return tuple(s == -1 for s in state)

    def _check(self, state, player):
        assert player == (len(state) - state.count(-1)) % self.n_player

    def human(self, state):
        board = tuple(zip_longest(*([iter(state)] * self.m)))
        return '\n'.join(' '.join('%+2d' % s for s in row) for row in board)

        
games = [Null, Binary, Flip, Count, Narrow, Matching, Roshambo, Modulo, Connect3, MNOP]

if __name__ == '__main__':
    from play import main  # noqa
    main()
