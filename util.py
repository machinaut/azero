#!/usr/bin/env python

import random
import numpy as np
from itertools import tee


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def softmax(x, mask=1):
    x = np.asarray(x)
    e = np.exp(x - x.max()) * mask
    s = e.sum()
    return e / s


def sample_logits(logits, valid=1, rs=np.random):
    probs = softmax(logits, valid)
    return rs.choice(range(len(probs)), p=probs)


def sample_probs(probs, rs=np.random):
    return rs.choice(range(len(probs)), p=probs)


def sample_games(games, rs=np.random):
    ''' Return (observation, probabilities, outcomes) arrays for training '''
    s = sum([[(o, q, z) for o, q in t] for t, z in games], [])
    d = [s[i] for i in rs.choice(len(s), len(games), replace=False)]
    return map(np.array, zip(*d))
