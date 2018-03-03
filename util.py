#!/usr/bin/env python

import numpy as np
from itertools import tee


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def view2obs(view, player):
    full_view = view + (player,)
    obs = np.array(full_view, dtype=float)
    return obs


def softmax(x, mask=1):
    x = np.asarray(x)
    e = np.exp(x - x.max()) * mask
    s = e.sum()
    return e / s


def sample_logits(logits, valid=1):
    probs = softmax(logits, valid)
    return np.random.choice(range(len(probs)), p=probs)


def sample_probs(probs):
    return np.random.choice(range(len(probs)), p=probs)
