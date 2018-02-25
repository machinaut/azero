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

    def test_xel2(self):
        rs = np.random.RandomState(0)
        A, B = 5, 4
        a = rs.rand(A)
        b = rs.rand(A)
        c = rs.rand(B)
        d = rs.rand(B)
        e = rs.rand(1)
        out, cache = nn.xel2_fwd(a, b, c, d, e)
        dout = rs.rand()
        da, db, dc, dd, de = nn.xel2_bak(dout, cache)
        na = finite_difference(lambda x: nn.xel2_fwd(x, b, c, d, e)[0], a, dout)
        nb = finite_difference(lambda x: nn.xel2_fwd(a, x, c, d, e)[0], b, dout)
        nc = finite_difference(lambda x: nn.xel2_fwd(a, b, x, d, e)[0], c, dout)
        nd = finite_difference(lambda x: nn.xel2_fwd(a, b, c, x, e)[0], d, dout)
        ne = finite_difference(lambda x: nn.xel2_fwd(a, b, c, d, x)[0], e, dout)
        np.testing.assert_allclose(da, na)
        np.testing.assert_allclose(db, nb)
        np.testing.assert_allclose(dc, nc)
        np.testing.assert_allclose(dd, nd)
        np.testing.assert_allclose(de, ne)


if __name__ == '__main__':
    unittest.main()
