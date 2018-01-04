#!/usr/bin/env python

from random import choice
from itertools import compress


def select(valid):
    ''' Select a random valid action '''
    return choice(list(compress(range(len(valid)), valid)))


def argmax(probs, valid):
    ''' Sample a weighted valid action '''
    return max(range(len(probs)), key=lambda x: probs[x] * valid[x])  # noqa
