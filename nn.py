#!/usr/bin/env python

import numpy as np


def relu_fwd(x):
    ''' rectified linear unit - forward pass '''
    out = np.maximum(x, 0)
    cache = (x,)
    return out, cache


def relu_bak(dout, cache):
    ''' rectified linear unit - backward pass '''
    x, = cache
    dx = dout * (x > 0)
    return dx


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
    xent = (p * q).sum(axis=1)
    l2 = np.square(d).sum(axis=1)
    out = c * xent + (1 - c) * l2
    cache = (p, q, d, xent, l2, c)
    return out, cache


def xel2_bak(dout, cache):
    ''' cross-entropy and L2 loss - backward pass '''
    p, q, d, xent, l2, c = cache
    dp = c * (q.T * dout.T).T
    dq = c * (p.T * dout.T).T
    doutd = (d.T * dout.T).T
    dv = 2 * (1 - c) * doutd
    dz = -2 * (1 - c) * doutd
    dc = dout.dot(xent - l2)
    return dp, dq, dv, dz, dc
