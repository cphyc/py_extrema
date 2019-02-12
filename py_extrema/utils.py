from numba import njit, guvectorize
from collections import namedtuple, OrderedDict
import numpy as np
import attr
import pandas as pd


class FiniteDictionary(OrderedDict):
    def __init__(self, maxlen, *args, **kwa):
        self._maxlen = maxlen
        super(FiniteDictionary, self).__init__(*args, **kwa)

    @property
    def maxlen(self):
        return self._maxlen

    @maxlen.setter
    def maxlen(self, v):
        if v < 0:
            raise Exception('Invalid maxlen %s', v)
        self._maxlen = v

    def __setitem__(self, k, v, *args, **kwa):
        if len(self) == self.maxlen and k not in self:
            # Remove oldest member
            self.popitem(False)

        super(FiniteDictionary, self).__setitem__(k, v, *args, **kwa)

    def last(self):
        return self[list(self.keys())[-1]]


@attr.s(frozen=True)
class CriticalPoints:
    pos = attr.ib(converter=np.atleast_2d)
    eigvals = attr.ib(converter=np.atleast_2d)
    kind = attr.ib(converter=np.atleast_1d)
    hessian = attr.ib(converter=np.atleast_3d)
    npt = attr.ib(converter=int)
    dens = attr.ib(converter=np.atleast_1d)

    def as_dataframe(self):
        x, y, z = self.pos.T
        l1, l2, l3 = self.eigvals.T
        h11, h22, h33, h12, h13, h23 = (
            self.hessian[:, i, j]
            for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))

        return pd.DataFrame(dict(
            x=x, y=y, z=z,
            l1=l1, l2=l2, l3=l3,
            kind=self.kind,
            h11=h11, h22=h22, h33=h33, h12=h12, h13=h13, h23=h23,
            dens=self.dens))


@njit
def unravel_index(index, shape):
    n = len(shape)
    result = np.zeros(n, dtype=np.int32)
    for i in range(n-1, -1, -1):
        s = shape[i]
        result[i] = index % s
        index //= s

    return result


@njit
def solve(A, B):
    '''Solve the equation A*X = B.'''

    N = A.shape[-1]
    if N == 2:
        a = A[..., 0, 0]
        b = A[..., 1, 1]
        c = A[..., 0, 1]

        b1 = B[..., 0]
        b2 = B[..., 1]

        det = a*b - c**2

        X = np.zeros(B.shape)
        X[..., 0] = (b*b1 - b2*c) / det
        X[..., 1] = (a*b2 - b1*c) / det
        
    elif N == 3:
        a = A[..., 0, 0]
        b = A[..., 1, 1]
        c = A[..., 2, 2]
        d = A[..., 0, 1]
        e = A[..., 0, 2]
        f = A[..., 1, 2]

        b1 = B[..., 0]
        b2 = B[..., 1]
        b3 = B[..., 2]

        d2 = d**2
        f2 = f**2
        e2 = e**2
        det = a*b*c - a*f2 - b*e2 - c*d2 + 2*d*e*f

        X = np.zeros(B.shape)
        X[..., 0] = (b*b1*c - b2*c*d - b*b3*e + b3*d*f + b2*e*f - b1*f2) / det
        X[..., 1] = (a*b2*c - b1*c*d + b3*d*e - b2*e2 - a*b3*f + b1*e*f) / det
        X[..., 2] = (a*b*b3 - b3*d2 - b*b1*e + b2*d*e - a*b2*f + b1*d*f) / det

        return X
