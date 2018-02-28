#!/usr/bin/env python

import unittest
import numpy as np
import nn


def multi_index_iterator(x):
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        yield it.multi_index
        it.iternext()


def finite_difference(f, x, df, h=1e-7):
    assert not np.issubdtype(x.dtype, np.integer)
    x = x.copy()
    x.setflags(write=True)
    grad = np.zeros_like(x)
    for ix in multi_index_iterator(x):
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    return grad


class TestNN(unittest.TestCase):

    def test_relu(self):
        rs = np.random.RandomState(0)
        A, B = 3, 4
        a = rs.rand(A, B)
        dout = rs.rand(A, B)
        out, cache = nn.relu_fwd(a)
        da = nn.relu_bak(dout, cache)
        na = finite_difference(lambda x: nn.relu_fwd(x)[0], a, dout)
        np.testing.assert_allclose(da, na)

    def test_mlp(self):
        rs = np.random.RandomState(0)
        A, B, C = 3, 4, 5
        x = rs.rand(A, B)
        W = rs.rand(B, C)
        b = rs.rand(C)
        out, cache = nn.mlp_fwd(x, W, b)
        dout = rs.rand(A, C)
        dx, dW, db = nn.mlp_bak(dout, cache)
        nx = finite_difference(lambda y: nn.mlp_fwd(y, W, b)[0], x, dout)
        nW = finite_difference(lambda y: nn.mlp_fwd(x, y, b)[0], W, dout)
        nb = finite_difference(lambda y: nn.mlp_fwd(x, W, y)[0], b, dout)
        np.testing.assert_allclose(dx, nx)
        np.testing.assert_allclose(dW, nW)
        np.testing.assert_allclose(db, nb)

    def test_loss(self):
        rs = np.random.RandomState(0)
        A, B, C = 5, 4, 3
        x = rs.randn(A, B)
        q = rs.randn(A, B)
        v = rs.randn(A, C)
        z = rs.randn(A, C)
        c = rs.randn(1)
        dout = rs.randn(A, 1)
        for arr in (x, q, v, z, c, dout):
            arr.setflags(write=False)
        out, cache = nn.loss_fwd(x, q, v, z, c)
        dx, dv = nn.loss_bak(dout, cache)
        nx = finite_difference(lambda y: nn.loss_fwd(y, q, v, z, c)[0], x, dout)
        nv = finite_difference(lambda y: nn.loss_fwd(x, q, y, z, c)[0], v, dout)
        np.testing.assert_allclose(dx, nx, atol=1e-6)
        np.testing.assert_allclose(dv, nv, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
