#!/usr/bin/env python

import numpy as np


def mlp_fwd(x, W, b):
    ''' multi-layer perceptron - forward pass '''
    out = x.dot(W) + b
    cache = (x, W, b)
    return out, cache


def mlp_bak(dout, cache):
    ''' multi-layer perceptron - backward pass '''
    x, W, b = cache
    dx = dout.dot(W.T)
    dW = x.T.dot(dout)
    db = dout.sum(axis=0)
    return dx, dW, db


def xel2_fwd(p, q, v, z, c):
    ''' cross-entropy and L2 loss - forward pass '''
    d = v - z
    xent = p.dot(q)
    l2 = np.square(d).sum()
    out = c * xent + (1 - c) * l2
    cache = (p, q, d, xent, l2, c)
    return out, cache


def xel2_bak(dout, cache):
    ''' cross-entropy and L2 loss - backward pass '''
    p, q, d, xent, l2, c = cache
    dp = dout * c * q
    dq = dout * c * p
    dv = 2 * dout * (1 - c) * d
    dz = -2 * dout * (1 - c) * d
    dc = dout * (xent - l2)
    return dp, dq, dv, dz, dc
