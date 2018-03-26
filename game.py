#!/usr/bin/env python

import random
import numpy as np
from itertools import zip_longest
''' for Nim  '''
from functools import reduce
from collections import defaultdict


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
            assert 0 <= player < self.n_player
            self._check(state, player)
        else:
            outcome = np.asarray(outcome, dtype=float)
            assert player is None
            assert outcome.size == self.n_player

    def _check(self, state, player):
        pass  # Optional: Implement in subclass

    def start(self):
        '''
        Start a new game, returns
            state - game state tuple
            player - index of the next player
            outcome - tuple or array of rewards or None if game is not over
        '''
        state, player, outcome = self._start()
        if outcome is not None:
            outcome = np.asarray(outcome, dtype=float)
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
            outcome - tuple or array of rewards or None if game is not over
        '''
        assert state is not None
        self.check(state, player)
        assert 0 <= action < self.n_action
        state, player, outcome = self._step(state, player, action)
        if outcome is not None:
            outcome = np.asarray(outcome, dtype=float)
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
        view = np.asarray(view, dtype=float)
        assert view.size == self.n_view
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
        State is a 5x4 numpy array, with -1 as empty, 0 as player 0,
        and 1 as player 1.'''
    n_action = 5
    n_state = 20
    n_player = 2
    poss = [([0, 0, 0], [0, -1, -2]),
            ([0, -1, -2], [0, -1, -2]),
            ([1, 0, -1], [1, 0, -1]),
            ([2, 1, 0], [2, 1, 0]),
            ([0, 1, 2], [0, -1, -2]),
            ([-1, 0, 1], [1, 0, -1]),
            ([-2, -1, 0], [2, 1, 0]),
            ([-2, -1, 0], [0, 0, 0, ]),
            ([-1, 0, 1], [0, 0, 0, ]),
            ([0, 1, 2], [0, 0, 0, ])]

    def _start(self):
        return np.ones((5, 4), dtype=np.int8) * -1, 0, None

    def _step(self, state, player, action):
        assert state[action, -1] == -1
        new_piece = np.where(state[action] == -1)[0][0]
        state[action, new_piece] = player
        # Check for victory
        # TODO: make this better
        for poss in self.poss:
            if self._win(state, player, action, new_piece, poss[0], poss[1]):
                return state, None, [(1, -1), (-1, 1)][player]
        # Check for tie
        if not np.any(state[:, -1] == -1):
            return state, None, (0, 0)
        # Game continues
        return state, 1 - player, None

    def _win(self, state, player, action, new_piece, x_set, y_set):
        for piece in range(3):
            if not 0 <= action + x_set[piece] < state.shape[0]:
                return False
            if not 0 <= new_piece + y_set[piece] < state.shape[1]:
                return False
            if (state[action + x_set[piece], new_piece + y_set[piece]] !=
                    player):
                return False
        return True

    def _valid(self, state, player):
        return state[:, -1] == -1

    def _view(self, state, player):
        return ()

    def _check(self, state, player):
        pass

    def human(self, state):
        buffer = [' '.join('%+2d' % s for s in row)
                  for row in state.transpose()]
        buffer.reverse()
        return '\n'.join(buffer)


class Nim(Game):
    ''' Nim, see https://en.wikipedia.org/wiki/Nim '''
    
    def __init__(self, s=(3,5,7), p=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ps = len(s)  # number of piles
        self.mp = max(s) # largest pile
        self.n_player = self.p = p
        self.n_action = self.ps * self.mp 
        self.n_state = self.n_view = self.ps * self.mp + 1
        self.st = (0,) * self.n_state 
        for i in range(self.ps):
            for j in range(s[i]):
                k = i * self.mp + j
                self.st = self.st[:k] + (1,) + self.st[k+1:]

    def _start(self):
        return self.st, 0, None

    def _step(self, state, player, action):
        assert state[action] == 1
        assert action < len(state) -1 # didn't play action = player variable
        cap = (int(action / self.mp) + 1) * self.mp # beginning index of the next pile
        # remove all the stones until the next pile
        i = action 
        while i < cap:
            state = state[:i] + (0,) + state[i + 1:]
            i += 1
        if self._win(state, player, action):
            outcome = [-1] * self.n_player
            outcome[player] = self.n_player - 1
            return None, None, tuple(outcome)
        nextplayer = (player + 1) % self.n_player
        return (state[:len(state)-1] + (nextplayer,)), nextplayer, None

    def _win(self, state, player, action):
        assert state[action] == 0 # Post state update
        if state[:len(state)-1].count(1) == 0:
            return True

    def _valid(self, state, player):
#        if state[len(state)-1] != player:
#            return tuple(False for s in state)
        return tuple(s == 1 for s in state[:len(state)-1]) # + (False,)

    def _check(self, state, player):
        assert state[len(state)-1] == player

    def human(self, state):
        st = state[:len(state)-1]
        board = tuple(zip_longest(*([iter(st)] * self.mp)))
        return '\n'.join(' '.join('%+2d' % s for s in row) for row in board)


class MNOP(Game):
    ''' Generalized tic-tac-toe '''

    def __init__(self, m=3, n=3, o=3, p=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert m >= o and n >= o  # Otherwise game is unwinnable
        self.m = m  # board width
        self.n = n  # board height
        self.o = o  # win length
        self.n_player = self.p = p
        self.n_action = self.n_state = m * n
        self.n_view = m * n * p

    def _start(self):
        return (-1,) * self.n_state, 0, None

    def _step(self, state, player, action):
        assert state[action] == -1
        state = state[:action] + (player,) + state[action + 1:]
        if self._win(state, player, action):
            outcome = -np.ones(self.n_player)
            outcome[player] = self.n_player - 1
            return state, None, outcome
        if state.count(-1) == 0:
            return state, None, (0,) * self.n_player
        return state, (player + 1) % self.n_player, None

    def _win(self, state, player, action):
        assert state[action] == player  # Post state update
        m, n, o, s = self.m, self.n, self.o, state  # Shorthand for short lines
        a, b = divmod(action, self.m)
        for i in range(max(0, a - o), min(a + o, m - o + 1)):
            if all(s[(i + k) * m + b] == player for k in range(o)):
                return True
        for j in range(max(0, b - o), min(b + o, n - o + 1)):
            if all(s[a * m + j + k] == player for k in range(o)):
                return True
        for l in range(max(-a, -b, 1 - o),
                       min(m - o - a, n - o - b, o - 1) + 1):
            if all(s[(a + l + k) * m + b + l + k] == player for k in range(o)):
                return True
        for l in range(max(1 - o, -a, b - n + 1),
                       min(o - 1, m - o - a, b - o + 1) + 1):
            if all(s[(a + l + k) * m + b - l - k] == player for k in range(o)):
                return True
        return False

    def _valid(self, state, player):
        return tuple(s == -1 for s in state)

    def _view(self, state, player):
        view = np.zeros((self.n_player, self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                k = state[i * self.m + j]
                if k >= 0:
                    p = (k - player) % self.n_player
                    view[p, i, j] = 1
        return view

    def _check(self, state, player):
        assert player == (len(state) - state.count(-1)) % self.n_player

    def human(self, state):
        str_state = (str(i) if i >= 0 else '-' for i in state)
        board = tuple(zip_longest(*([iter(str_state)] * self.m)))
        return '\n'.join(' '.join(row) for row in board)
        
class Checkers(Game):
    ''' Checkers with multiple board sizes. '''
    
    def __init__(self, size=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert size % 2 == 0 # Only allowing even checkers boards
        self.size = size
        self.n_player = 2
        self.board_size = size * size // 2
        self.n_action = self.board_size * 4
        self.n_view = self.board_size * 5
        self.n_state = self.board_size + 3
        self.max_move = self.size ** 3
        self.moves_fwd = defaultdict(list)
        self.jumps_fwd = defaultdict(list)
        self.moves_bak = defaultdict(list)
        self.jumps_bak = defaultdict(list)
        row_size = self.size // 2
        for loc in range(self.board_size):
            row, col = divmod(loc, row_size)
            #moves
            if row % 2 == 0: # and row < self.size-1
                #moves_fwd
                if col == 0: #can always move forward if row is even
                    self.moves_fwd[loc].append((self.board_size+loc,loc+row_size))
                else:
                    self.moves_fwd[loc].append((self.board_size+loc,loc+row_size))
                    self.moves_fwd[loc].append((loc,loc+row_size-1))
                #moves_bak
                if row > 0:
                    if col == 0:
                        self.moves_bak[loc].append((self.board_size*3+loc,loc-row_size))
                    else:
                        self.moves_bak[loc].append((self.board_size*3+loc,loc-row_size))
                        self.moves_bak[loc].append((self.board_size*2+loc,loc-row_size-1))
            if row % 2 == 1: # and row < self.size-1
                #moves_fwd
                if row < self.size - 1:
                    if col == row_size - 1:
                        self.moves_fwd[loc].append((loc,loc+row_size))
                    else:
                        self.moves_fwd[loc].append((loc,loc+row_size))
                        self.moves_fwd[loc].append((self.board_size+loc,loc+row_size+1))
                #moves_bak
                if col == row_size - 1: #can always move back if row is odd
                    self.moves_bak[loc].append((self.board_size*2+loc,loc-row_size))
                else:
                    self.moves_bak[loc].append((self.board_size*2+loc,loc-row_size))
                    self.moves_bak[loc].append((self.board_size*3+loc,loc-row_size+1))
            #jumps_fwd
            if row < self.size - 2:
                if col > 0:
                    self.jumps_fwd[loc].append((loc,loc+row_size - 1 + row % 2,loc+2*row_size-1))
                if col < row_size - 1:
                    self.jumps_fwd[loc].append((self.board_size + loc, loc + row_size + row % 2, loc + 2 * row_size + 1))
            #jumps_bak
            if row > 1:
                if col > 0:
                    self.jumps_bak[loc].append((2 * self.board_size + loc, loc - row_size - 1 + row % 2, loc - 2 * row_size - 1))
                if col < row_size - 1:
                    self.jumps_bak[loc].append((3 * self.board_size + loc, loc - row_size + row % 2, loc - 2 * row_size + 1))

    def _start(self):
        num_pieces = self.size // 2 * ( self.size // 2 - 1 )
        return (1,) * num_pieces + (0,)* self.size + (-1,) * num_pieces + (0, 0, -1), 0, None

    def _step(self, state, player, action):
        state = list(state)
        state[-3] += 1
        if state[-3] == self.max_move:
            return (tuple(state), None, [0, 0])
        move, piece = divmod(action, self.board_size)
        poss_actions = []
        if player == 1:
            move = 3 - move
            piece = self.board_size - piece - 1
            #action = move * self.board_size + piece
        if player == 0 or abs(state[piece]) == 2:
            poss_actions.extend([x[0] for x in self.moves_fwd[piece]])
            poss_actions.extend([x[0] for x in self.jumps_fwd[piece]])
        if player == 1 or abs(state[piece]) == 2:
            poss_actions.extend([x[0] for x in self.jumps_bak[piece]])
            poss_actions.extend([x[0] for x in self.moves_bak[piece]])
        #valid = self.valid(state, player)
        #sparse = tuple(i for i in range(len(valid)) if valid[i])
        #assert action in sparse
        #assert action in poss_actions, "{} {} {} {}".format(action, poss_actions, sparse, state)
        #if action not in poss_actions:
        #    import ipdb; ipdb.set_trace()
        enemies = [-1, -2] if player == 0 else [1, 2]
        friendlies = [-1, -2] if player == 1 else [1, 2]
        assert state[piece] in friendlies
        row_size = self.size // 2
        row, col = divmod(piece, row_size)
        dx = move % 2 + row % 2 - 1 
        dy = row_size if move < 2 else -row_size
        dest = piece + dx + dy
        if state[dest] == 0: #Move
            state[dest] = state[piece] # Does it get kinged?
            if (row == 1 and player == 1) or (row == self.size - 2 and player == 0):
                state[dest] = friendlies[-1]
            state[piece] = 0
            state[-1] = -1
            state[-2] = (player + 1) % 2
            return (tuple(state), state[-2], None)
        assert state[dest] in enemies #Jump
        state[dest] = 0
        land = piece + 2 * dy + (move % 2) * 2 - 1
        assert state[land] == 0
        state[land] = state[piece]
        state[piece] = 0
        #Can you jump again?
        can_jump = False
        if player == 0 or abs(state[land]) == 2:
            for _, enemy_idx, land_idx in self.jumps_fwd[land]:
                if state[enemy_idx] in enemies and state[land_idx] == 0:
                    can_jump = True
                    break
        if not can_jump and (player == 1 or abs(state[land]) == 2):
            for _, enemy_idx, land_idx in self.jumps_bak[land]:
                if state[enemy_idx] in enemies and state[land_idx] == 0:
                    can_jump = True
                    break
        if can_jump:
            state[-1] = land
            state[-2] = player
            return (tuple(state), state[-2], None)
        #Did you promote?
        if (row == 2 and player == 1) or (row == self.size - 3 and player == 0):
            state[land] = friendlies[-1]
        #Did you win?
        enemy_count = 0
        for enemy in enemies:
            enemy_count += state[:-2].count(enemy)
        if enemy_count == 0:
            return (tuple(state), None, [[1, -1], [-1, 1]][player])
        state[-1] = -1
        state[-2] = (player + 1) % 2
        return (tuple(state), state[-2], None)
        
        

    def _valid(self, state, player):
        actions = [False] * self.n_action
        board = state[:-3] if player == 0 else state[:-3][::-1]
        
        enemies = [-1, -2] if player == 0 else [1, 2]
        friendlies = [-1, -2] if player == 1 else [1, 2]
        if state[-1] != -1:
            piece = state[-1] if player == 0 else self.board_size - state[-1] - 1
            for action_idx, enemy_idx, land_idx in self.jumps_fwd[piece]:
                if board[land_idx] == 0 and board[enemy_idx] in enemies:
                    actions[action_idx] = True
            if abs(board[piece]) == 2:            
                for action_idx, enemy_idx, land_idx in self.jumps_bak[piece]:
                    if board[land_idx] == 0 and board[enemy_idx] in enemies:
                        actions[action_idx] = True            
        else:
            for (idx, piece) in enumerate(board):
                if piece in friendlies:
                    for action_idx, enemy_idx, land_idx in self.jumps_fwd[idx]:
                        if board[land_idx] == 0 and board[enemy_idx] in enemies:
                            actions[action_idx] = True
                    if abs(piece) == 2:     
                        for action_idx, enemy_idx, land_idx in self.jumps_bak[idx]:
                            if board[land_idx] == 0 and board[enemy_idx] in enemies:
                                actions[action_idx] = True
            if sum(actions) == 0:
                for (idx, piece) in enumerate(board):
                    if piece in friendlies:
                        for action_idx, land_idx in self.moves_fwd[idx]:
                            if board[land_idx] == 0:
                                actions[action_idx] = True
                        if abs(piece) == 2:                    
                            for action_idx, land_idx in self.moves_bak[idx]:
                                if board[land_idx] == 0:
                                    actions[action_idx] = True
        return tuple(actions)

    def _view(self, state, player):
        view = np.zeros((5, self.board_size))
        if player == 0:
            for i, piece_type in enumerate([1, 2, -1, -2]):
                for j, piece in enumerate(state[:-3]):
                    view[i, j] = piece == piece_type
            if state[-1] != -1:
                view[4,state[-1]] = 1
        else:
            for i, piece_type in enumerate([-1, -2, 1, 2]):
                for j, piece in enumerate(state[:-3]):
                    view[i, self.board_size - j - 1] = piece == piece_type
            if state[-1] != -1:
                view[4,self.board_size - state[-1] - 1] = 1
        return view
        
    def _check(self, state, player):
        assert player == state[-2]

    def human(self, state, player):
        display = ['X', 'x', '-', 'o', 'O'] if player == 0 else ['O', 'o', '-', 'x', 'X']
        board = state[:-3] if player == 0 else state[:-3][::-1]
        str_state = [display[i+2] for i in board]
        s = '*'+self.size*'-'+'*\n'
        for row in range(self.size):
            s += '|' if row % 2 == 0 else '| '
            r = []
            for col in range(self.size // 2):
                loc = row*(self.size // 2) + col
                r.append(str_state[loc])
            s += ' '.join(r)
            s += ' |\n' if row % 2 == 0 else '|\n'
        s += '*'+self.size*'-'+'*\n'
        return s

games = [Null, Binary, Flip, Count, Narrow,
         Matching, Roshambo, Modulo, MNOP, Checkers]

if __name__ == '__main__':
    from play import main  # noqa
    main()
