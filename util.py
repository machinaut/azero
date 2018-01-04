#!/usr/bin/env python

from random import choice
from itertools import compress


def select(valid):
    ''' Select a random valid action '''
    return choice(list(compress(range(len(valid)), valid)))
