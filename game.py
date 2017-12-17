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

Rows are called ranks
Black's ranks are 1-4, 5-8, 9-12
White's ranks are 21-24, 25-28, 29-32
'''


from array import array


def square2rank(square):
    return ((square - 1) // 4) + 1


class BoardState:
    def __init__(self):
        self.state = array('b', 32)
