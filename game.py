#!/usr/bin/env python

import random


class Game:
    ''' Interface class for a game to be optimized by alphazero algorithm '''
    n_action = 0  # Number of possible actions
    n_state = 0  # Size of the state tuple
    n_player = 1  # Number of players
    n_view = 0  # Size of the observation of each player

    def __init__(self, seed=None) -> None:
        self.random = random.Random(seed)

    def start(self):
        '''
        Start a new game, returns
            state - game state tuple
            player - index of the next player
            outcome - index of winning player or None if game is not over
        '''
        state, player, outcome = self._start()
        self._check(state, player)
        assert len(state) == self.n_state
        if outcome is None:
            assert 0 <= player < self.n_action
        else:
            assert player == -1
            assert outcome < self.n_player
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
        assert len(state) == self.n_state
        assert 0 <= player < self.n_player
        assert 0 <= action < self.n_action
        self._check(state, player)
        state, player, outcome = self._step(state, player, action)
        assert len(state) == self.n_state
        if outcome is None:
            assert 0 <= player < self.n_action
            self._check(state, player)
        else:
            assert player == -1
            assert outcome < self.n_player
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
        assert len(state) == self.n_state
        assert 0 <= player < self.n_player
        self._check(state, player)
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
        assert len(state) == self.n_state
        assert 0 <= player < self.n_player
        self._check(state, player)
        view = self._view(state, player)
        assert len(view) == self.n_view
        return view

    def _view(self, state, player):
        return state  # default to full visibility

    def _check(self, state, player):
        '''
        Check if a combination of state and player are valid.
        Raises AssertionError() if not!
        '''
        pass

    def human(self, state):
        ''' Print out a human-readable state '''
        return str(state)


class Null(Game):
    ''' Null game, always lose '''
    def _start(self):
        return (), -1, -1

    def _valid(self, state, player):
        assert False


class Binary(Game):
    ''' Single move game, 0 - loses, 1 - wins '''
    n_action = 2

    def _start(self):
        return (), 0, None

    def _step(self, state, player, action):
        outcome = 0 if action == 1 else -1
        return (), -1, outcome

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
        outcome = 0 if action == state[0] else -1
        return (-1,), -1, outcome

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
            return (-1,), -1, -1  # A loser is you
        if action == count == 2:
            return (-1,), -1, 0  # A winner is you
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
            return (-1,), -1, -1
        return (action,), 0, None

    def _valid(self, state, player):
        return tuple(i < state[0] for i in range(3))

    def _check(self, state, player):
        assert 0 <= state[0] <= 3


class Matching(Game):
    ''' Matching Pennies '''
    n_action = 2
    n_state = 1
    n_player = 2

    def _start(self):
        return (-1,), 0, None

    def _step(self, state, player, action):
        if state[0] == -1:
            return (action,), 1, None
        else:
            outcome = 0 if state[0] == action else 1
            return (2,), -1, outcome

    def _valid(self, state, player):
        return (True, True)

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        assert -1 <= state[0] < 2
        if state[0] == -1:
            assert player == 0
        else:
            assert player == 1


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
        else:
            if (action - 1) % 3 == roshambo:
                return (3, -1), -1, 0  # First player wins
            elif (action + 1) % 3 == roshambo:
                return (3, -1), -1, 1  # Second player wins
            else:
                assert action == roshambo
                return (3, -1), -1, -1  # Tie, both lose

    def _valid(self, state, player):
        return (True, True, True)

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        roshambo, current = state
        assert -1 <= roshambo < 3
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
        else:
            outcome = total % 3
            return (-1, -1), -1, outcome

    def _valid(self, state, player):
        return (True, True, True)

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        total, current = state
        assert 0 <= total < 6
        assert player == current


games = [Null, Binary, Flip, Count, Narrow, Matching, Roshambo, Modulo]


if __name__ == '__main__':
    from play import main  # noqa
    main()
