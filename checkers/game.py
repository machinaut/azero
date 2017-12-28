#!/usr/bin/env python

'''
Compact representation:

Row     Column
    0 1 2 3 4 5 6 7
 0  R   R   R   R
 1    R   R   R   R
 2  R   R   R   R
 3    -   -   -   -
 4  -   -   -   -
 5    B   B   B   B
 6  B   B   B   B
 7    B   B   B   B

Squares:
       1   2   3   4
     5   6   7   8
       9  10  11  12
    13  14  15  16
      17  18  19  20
    21  22  23  24
      25  26  27  28
    29  30  31  32
(these are indexed 0-31, which is confusing, I know)

Rows are called ranks
Black's ranks are 1-4, 5-8, 9-12
White's ranks are 21-24, 25-28, 29-32
'''


from array import array


def square2rank(square):
    return ((square - 1) // 4) + 1


BLACK_PAWN_VALID = {}
WHITE_PAWN_VALID = {}
BLACK_KING_VALID = {}
WHITE_KING_VALID = {}
for rank in range(0, 8, 2):  # Even ranks
    pass
for rank in range(1, 8, 2):  # Odd ranks
    pass




class BoardState:
    def __init__(self, start=None):
        # State array, -1 is Black, +1 is White (2 for kings)
        self.state = array('b', 32)
        if start is None:
            for i in range(12):
                self.state[i] = -1  # Black starting ranks
                self.state[i + 16] = 1  # White starting ranks

    def square(self, square):
        return self.state[square - 1]

    def valid(self, start, end):
        # JUST USE A LOOKUP TABLE AND CALL IT A FREAKING DAY
        # TODO: include whos turn it is?  Can't move other pieces
        color = self.square(start)
        if color == 0 or self.square(end) != 0:
            return False  # No piece at start location or end is occupied
        elif color == -1:
            # Black pawns go down
            pass
