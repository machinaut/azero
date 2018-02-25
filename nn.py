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
