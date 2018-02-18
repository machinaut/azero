#!/usr/bin/env python

import random


class Game:
    ''' Interface class for a game to be optimized by alphazero algorithm '''
    n_action = 0
    n_state = 0
    n_player = 1

    def __init__(self, seed=None) -> None:
        self.random = random.Random(seed)

    def start(self):
        '''
        Start a new game, returns
            state - game state object (can be None)
            player - index of the next player
            outcome - index of winning player or None if game is not over
        '''
        state, player, outcome = self._start()
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
        state, player, outcome = self._step(state, player, action)
        assert len(state) == self.n_state
        if outcome is None:
            assert 0 <= player < self.n_action
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
        return self._view(state, player)

    def _view(self, state, player):
        return state  # default to full visibility

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
    n_state = 1
    n_player = 1

    def _start(self):
        return (0,), 0, None

    def _step(self, state, player, action):
        assert state == (0,)
        assert 0 <= action < 2
        outcome = 0 if action == 1 else -1
        return (-1,), -1, outcome

    def _valid(self, state, player):
        assert state == (0,)
        return (True, True)


class Flip(Game):
    ''' Guess a coin flip '''
    n_action = 2
    n_state = 1
    n_player = 1

    def _start(self):
        coin = self.random.randrange(2)
        return (coin,), 0, None

    def _step(self, state, player, action):
        assert len(state) == 1
        assert 0 <= state[0] < 2
        assert 0 <= action < 2
        outcome = 0 if action == state[0] else -1
        return (-1,), -1, outcome

    def _valid(self, state, player):
        assert 0 <= state[0] < 2
        return (True, True)


class Count(Game):
    ''' Count to 3 '''
    n_action = 3
    n_state = 1
    n_player = 1

    def _start(self):
        return (0,), 0, None

    def _step(self, state, player, action):
        count, = state
        assert 0 <= count < 3
        if action != count:
            return (-1,), -1, -1  # A loser is you
        if action == count == 2:
            return (-1,), -1, 0  # A winner is you
        return (count + 1,), 0, None

    def _valid(self, state, player):
        count, = state
        assert 0 <= count < 3
        return (True, True, True)


class Narrow(Game):
    ''' Fewer choices every step '''
    n_action = 3
    n_state = 1
    n_player = 1

    def _start(self):
        return (3,), 0, None

    def _step(self, state, player, action):
        assert 0 <= state[0] <= 3
        assert 0 <= action < state[0]
        if action == 0:
            return (-1,), -1, -1
        return (action,), 0, None

    def _valid(self, state, player):
        assert 0 <= state[0] <= 3
        return tuple(i < state[0] for i in range(3))


class Matching(Game):
    ''' Matching Pennies '''
    n_action = 2
    n_state = 1
    n_player = 2

    def _start(self):
        return (-1,), 0, None

    def _step(self, state, player, action):
        assert -1 <= state[0] < 2
        if state[0] == -1:
            assert player == 0
            return (action,), 1, None
        else:
            assert player == 1
            outcome = 0 if state[0] == action else 1
            return (2,), -1, outcome

    def _valid(self, state, player):
        assert -1 <= state[0] < 2
        if state[0] == -1:
            assert player == 0
        else:
            assert player == 1
        return (True, True)


class Roshambo(Game):
    ''' Rock Paper Scissors '''
    n_action = 3
    n_state = 2
    n_player = 2

    def _start(self):
        return (0, 0), 0, None

    def _step(self, state, player, action):
        roshambo, current = state
        assert 0 <= roshambo < 3
        assert 0 <= current < 2
        assert current == player
        if current == 0:
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
        roshambo, current = state
        assert -1 <= roshambo < 3
        assert current == player
        return (True, True, True)


class Modulo(Game):
    ''' player mod 3 '''
    n_action = 3
    n_state = 2
    n_player = 3

    def _start(self):
        return (0, 0), 0, None

    def _step(self, state, player, action):
        total, current = state
        assert 0 <= total < 6
        assert player == current
        total += action
        current += 1
        if player < 2:
            return (total, current), current, None
        else:
            outcome = total % 3
            return (-1, -1), -1, outcome

    def _valid(self, state, player):
        total, current = state
        assert 0 <= total < 6
        assert player == current
        return (True, True, True)


games = [Null, Binary, Flip, Count, Narrow, Matching, Roshambo, Modulo]


if __name__ == '__main__':
    from play import main  # noqa
    main()
