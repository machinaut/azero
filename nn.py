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


def loss_fwd(x, q, z, c):
    ''' softmax cross-entropy and mean-squared-error combination - forward '''
    _, P = q.shape
    logits = x[:, :P] - np.max(x[:, :P], axis=1, keepdims=True)
    e = np.exp(logits)
    Z = np.sum(e, axis=1, keepdims=True)
    d = x[:, P:] - z
    out = (c * np.sum((logits - np.log(Z)) * q, axis=1, keepdims=True) +
           (1 - c) * np.sum(np.square(d), axis=1, keepdims=True))
    cache = (q, e, Z, d, c)
    return out, cache


def loss_bak(dout, cache):
    ''' softmax cross-entropy and mean-squared-error combination - backward '''
    q, e, Z, d, c = cache
    D, P = q.shape
    dx = np.empty((D, P + d.shape[1]))
    dx[:, P:] = 2 * d * (1 - c) * dout
    dx[:, :P] = c * dout * q
    dx[:, :P] -= e * (np.sum(dx[:, :P], axis=1, keepdims=True) / Z)
    return dx
