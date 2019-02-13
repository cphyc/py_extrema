from numba import njit, jit, guvectorize
from collections import OrderedDict
import numpy as np
import numexpr as ne
import attr
import pandas as pd
from itertools import product
from scipy.interpolate import interpn


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


def solve(A, B):
    '''Solve the equation A*X = B.'''

    N = A.shape[-1]
    if N == 2:
        a = A[..., 0, 0]
        b = A[..., 1, 1]
        c = A[..., 0, 1]

        b1 = B[..., 0]
        b2 = B[..., 1]

        det = ne.evaluate('a*b - c**2')

        X = np.zeros(B.shape, order='F')
        X[..., 0] = ne.evaluate('(b*b1 - b2*c) / det')
        X[..., 1] = ne.evaluate('(a*b2 - b1*c) / det')

        return X

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

        det = ne.evaluate('a*b*c - a*f2 - b*e2 - c*d2 + 2*d*e*f')

        X = np.zeros(B.shape, order='F')
        X[..., 0] = ne.evaluate('(b*b1*c - b2*c*d - b*b3*e + b3*d*f + b2*e*f - b1*f2) / det')
        X[..., 1] = ne.evaluate('(a*b2*c - b1*c*d + b3*d*e - b2*e2 - a*b3*f + b1*e*f) / det')
        X[..., 2] = ne.evaluate('(a*b*b3 - b3*d2 - b*b1*e + b2*d*e - a*b2*f + b1*d*f) / det')

        return X
    else:
        return np.linalg.solve(A, B)


@guvectorize(['void(float64[:], float64[:,:,:,:], float64[:])'],
             '(N),(M,i,i,i)->(M)')
def trilinear_interpolation(pos, v, ret):
    '''Compute the trilinear interpolation of data at given position

    Arguments
    ---------
    pos : (Ndim, ) float array
       The position w.r.t to the lower left edget of the cube.
    v : (M, 2, 2, 2) float array
       The value at the edges.

    Returns
    -------
    interp : float
       The interpolated value.

    Notes
    -----
    The original code is from
    http://paulbourke.net/miscellaneous/interpolation/

    '''
    xl, yl, zl = pos
    xr, yr, zr = 1-pos

    # Note the (inverse) order here!
    x = (xr, xl)
    y = (yr, yl)
    z = (zr, zl)
    ret[...] = 0

    for i in range(2):
        for j in range(2):
            for k in range(2):
                vol = x[i] * y[j] * z[k]
                ret[...] += v[..., i, j, k] * vol


@njit
def gradient(A, axis, dx=1):
    out = np.zeros_like(A)

    ijk = np.array([0, 0, 0], dtype=np.int32)
    ijkl = np.array([0, 0, 0], dtype=np.int32)
    ijkr = np.array([0, 0, 0], dtype=np.int32)

    i0 = j0 = k0 = 0
    iN, jN, kN = A.shape

    if axis == 0:
        i0 += 1
        iN -= 1
    elif axis == 1:
        j0 += 1
        jN -= 1
    elif axis == 2:
        k0 += 1
        kN -= 1

    for i in range(i0, iN):
        ijk[0] = ijkl[0] = ijkr[0] = i
        if axis == 0:
            ijkl[0] -= 1
            ijkr[0] += 1
        for j in range(j0, jN):
            ijk[1] = ijkl[1] = ijkr[1] = j
            if axis == 1:
                ijkl[1] -= 1
                ijkr[1] += 1
            for k in range(k0, kN):
                ijk[2] = ijkl[2] = ijkr[2] = k
                if axis == 2:
                    ijkl[2] -= 1
                    ijkr[2] += 1

                out[i, j, k] = (A[ijkr[0], ijkr[1], ijkr[2]] - A[ijkl[0], ijkl[1], ijkl[2]]) / 2 / dx

    # Left edge
    if axis == 0:
        i0 = 0
        iN = 1
    elif axis == 1:
        j0 = 0
        jN = 1
    elif axis == 2:
        k0 = 0
        kN = 1

    for i in range(i0, iN):
        ijk[0] = ijkr[0] = i
        if axis == 0:
            ijkr[0] += 1
        for j in range(j0, jN):
            ijk[1] = ijkr[1] = j
            if axis == 1:
                ijkr[1] += 1
            for k in range(k0, kN):
                ijk[2] = ijkr[2] = k
                if axis == 2:
                    ijkr[2] += 1

                out[i, j, k] = (A[ijkr[0], ijkr[1], ijkr[2]] - A[ijk[0], ijk[1], ijk[2]]) / dx

    # Right edge
    if axis == 0:
        i0 = A.shape[0]-1
        iN = A.shape[0]
    elif axis == 1:
        j0 = A.shape[1]-1
        jN = A.shape[1]
    elif axis == 2:
        k0 = A.shape[2]-1
        kN = A.shape[2]

    for i in range(i0, iN):
        ijk[0] = ijkl[0] = i
        if axis == 0:
            ijkl[0] -= 1
        for j in range(j0, jN):
            ijk[1] = ijkl[1] = j
            if axis == 1:
                ijkl[1] -= 1
            for k in range(k0, kN):
                ijk[2] = ijkl[2] = k
                if axis == 2:
                    ijkl[2] -= 1

                out[i, j, k] = (A[ijk[0], ijk[1], ijk[2]] - A[ijkl[0], ijkl[1], ijkl[2]]) / dx

    return out


@jit(looplift=True)
def measure_hessian(position, data, LE=np.array([0, 0, 0])):
    '''Compute the value of the hessian of the field at the given position.

    Arguments
    ---------
    position : ndarray (Npt, Ndim)
       The position of the points in space in pixel units.
    data : ndarray (Npt, Npt, Npt)
       The field itself
    '''
    LE = np.asarray(LE)
    Npt = len(position)
    N = data.shape[0]

    buff = np.empty((6, 6, 6))
    # Contains the value of h_ij at the corner
    hij_buff = np.empty((6, 2, 2, 2))
    tmp_buff = np.empty((6, 6, 6))
    ret = np.empty((Npt, 3, 3))

    ipos = np.empty(3, dtype=np.int32)
    jpos = np.empty(3, dtype=np.int32)
    dpos = np.empty(3, dtype=np.float64)

    for ipt in range(Npt):
        pos = position[ipt] - LE

        ipos[:] = pos-2
        jpos[:] = ipos+6
        dpos[:] = pos - ipos - 2

        # Copy data with periodic boundaries
        for i0, i in enumerate(range(ipos[0], jpos[0])):
            for j0, j in enumerate(range(ipos[1], jpos[1])):
                for k0, k in enumerate(range(ipos[2], jpos[2])):
                    buff[i0, j0, k0] = data[i % N, j % N, k % N]

        # Compute hessian using finite difference
        ii = 0
        for idim in range(3):
            for jdim in range(idim+1):
                tmp_buff[:] = gradient(gradient(buff, axis=idim), axis=jdim)
                hij_buff[ii, :, :, :] = tmp_buff[2:4, 2:4, 2:4]

                ii += 1

        # Perform trilinear interpolation of the hessian
        tmp = trilinear_interpolation(dpos, hij_buff)
        ii = 0
        for idim in range(3):
            for jdim in range(idim+1):
                ret[ipt, idim, jdim] = \
                  ret[ipt, jdim, idim] = tmp[ii]
                ii += 1

    return ret

@jit(looplift=True)
def measure_gradient(position, data, LE=np.array([0, 0, 0])):
    '''Compute the value of the gradient of the field at the given position.

    Arguments
    ---------
    position : ndarray (Npt, Ndim)
       The position of the points in space in pixel units.
    data : ndarray (Npt, Npt, Npt)
       The field itself
    '''
    LE = np.asarray(LE)
    Npt = len(position)
    N = data.shape[0]

    buff = np.empty((4, 4, 4))
    # Contains the value of h_ij at the corner
    grad_buff = np.empty((3, 2, 2, 2))
    tmp_buff = np.empty((4, 4, 4))
    ret = np.empty((Npt, 3))

    ipos = np.empty(3, dtype=np.int32)
    jpos = np.empty(3, dtype=np.int32)
    dpos = np.empty(3, dtype=np.float64)

    for ipt in range(Npt):
        pos = position[ipt] - LE

        ipos[:] = pos-1
        jpos[:] = ipos+4
        dpos[:] = pos - ipos - 1

        # Copy data with periodic boundaries
        for i0, i in enumerate(range(ipos[0], jpos[0])):
            for j0, j in enumerate(range(ipos[1], jpos[1])):
                for k0, k in enumerate(range(ipos[2], jpos[2])):
                    buff[i0, j0, k0] = data[i % N, j % N, k % N]

        # Compute hessian using finite difference
        ii = 0
        for idim in range(3):
            tmp_buff[:] = gradient(buff, axis=idim)
            grad_buff[ii, :, :, :] = tmp_buff[1:3, 1:3, 1:3]

            ii += 1

        # Perform trilinear interpolation of the hessian
        tmp = trilinear_interpolation(dpos, grad_buff)
        ii = 0
        for idim in range(3):
            ret[ipt, idim] = tmp[ii]
            ii += 1

    return ret


def measure_third_derivative(position, data, eigenvectors):
    '''Measure the third derivative in the eigenframe

    Arguments
    ---------
    position : (N, Ndim)
    data : (M, M, M)
    eigenvectors : (N, Ndim, Ndim)

    Returns
    -------
    Fxii : (N, Ndim)
       The interpolated value of the derivative at the position.'''
    N, Ndim = position.shape
    M = eigenvectors.shape[0]

    if eigenvectors.shape != (N, Ndim, Ndim):
        raise Exception(f'Wrong shape for eigenvectors: {eigenvectors.shape}, expected {(N, Ndim, Ndim)}.')
    if Ndim != 3:
        raise Exception(f'Got dimension {Ndim}, expected Ndim=3.')

    e1 = eigenvectors[..., 0]
    e2 = eigenvectors[..., 1]
    e3 = eigenvectors[..., 2]

    # Extend the field for interpolation
    F = data
    M = F.shape[0]
    Fext = np.zeros((M+2, M+2, M+2))
    x0 = position

    # Interpolate field along three eigenvectors
    dx = np.arange(-2, 3)

    x1 = (x0[:, None, :] + dx[None, :, None] * e1[:, None, :]) % M
    x2 = (x0[:, None, :] + dx[None, :, None] * e2[:, None, :]) % M
    x3 = (x0[:, None, :] + dx[None, :, None] * e3[:, None, :]) % M

    # Fill array for interpolation
    xi = np.zeros((3, N, 5, Ndim))
    xi[0, ...] = x1
    xi[1, ...] = x2
    xi[2, ...] = x3

    # Extend field on boundaries for linear interpolation near edges
    sl1 = (slice(1, -1), slice(None))
    sl2 = (slice(1), slice(-1, None))
    sl3 = (slice(-1, None), slice(1))

    sl_all = (sl1, sl2, sl3)

    for (sx1, sx2), (sy1, sy2), (sz1, sz2) in product(*[sl_all]*3):
        Fext[tuple((sx1, sy1, sz1))] = F[tuple((sx2, sy2, sz2))]
    xx = np.arange(-1, M+1)

    # Interpolate third derivative using second order finite
    # difference. Values has shape (3, Npt, 5)
    values = interpn((xx, xx, xx), Fext, xi)
    coeff = np.array([-1/2, 1, 0, -1, 1/2])
    Fxii = np.dot(values, coeff)  # shape: (3, Npt)

    return Fxii.T
