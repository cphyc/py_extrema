import numpy as np
cimport numpy as np
from numba import jit
import itertools
cimport cython
from cython.parallel import parallel, prange
from time import time

from cython.view cimport array as cvarray

cdef double start_time = time()

def my_print(str s):
    print(time() - start_time, s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef compute_extrema(double[:, :, :] data):
    cdef int N, M, L, count
    cdef int[6] I, J, K
    # cdef int[:, :] AA_order, AA_factor
    cdef double[:, :, :, :] MM, A, rhs
    cdef double[:, :, :, :, :] AA
    cdef double[:, :, :] chunk = np.zeros((3, 3, 3), dtype=np.float64),
    cdef double[:, :, :] tmp_chunk


    cdef np.int16_t[:, :, :] mask
    cdef double pos[3]
    cdef int i, j, k, l, ii, jj, kk
    cdef double[3] xx, cell_center
    cdef np.ndarray[np.float64_t, ndim=3] X, Y, Z, X2, Y2, Z2, XY, XZ, YZ
    cdef np.ndarray[np.float64_t, ndim=2] XX, XXinv
    cdef np.ndarray[long, ndim=1] tmp
    cdef np.ndarray[np.float64_t, ndim=3] AA_small
    cdef np.ndarray[np.float64_t, ndim=4] xyz
    cdef np.ndarray[np.float64_t, ndim=2] xyz_small, eigvals

    cdef np.ndarray[np.int64_t, ndim=1] II, JJ, KK, II0, JJ0, KK0
    my_print('Entering')
    # Computing common coefficients
    xx = [-1, 0, 1]
    X, Y, Z = np.meshgrid(xx, xx, xx, indexing='ij')
    X2 = X**2
    Y2 = Y**2
    Z2 = Z**2

    # Get the indices of the neighboring cells
    ii = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                tmp = np.array([i, j, k], dtype=long)
                # Only keep direct neighbors
                if (tmp == 0).sum() == 2:
                    I[ii] = i
                    J[ii] = j
                    K[ii] = k
                    ii += 1

    # Build XX matrix
    XX = np.zeros((9, 9), dtype=np.float64)
    XX[:, :6] = np.array([X2[I, J, K], Y2[I, J, K], Z2[I, J, K],
                          X[I, J, K], Y[I, J, K], Z[I, J, K],
                          (X*Y)[I, J, K], (X*Z)[I, J, K], (Y*Z)[I, J, K]])

    for i in range(6, 9):
        XX[i, i] = 1

    XXinv = np.linalg.inv(XX)

    N = data.shape[0]
    M = data.shape[1]
    L = data.shape[2]

    my_print('Computed constant arrays')
    # Compute MM matrix
    MM = np.empty((9, N, M, L), dtype=np.float64)
    # II0 = np.arange(0, N, dtype=np.int)
    # JJ0 = np.arange(0, M, dtype=np.int)
    # KK0 = np.arange(0, L, dtype=np.int)

    # l = 6
    # for ii in range(6):
    #     II = np.mod(II0 + I[ii] - 1, N)
    #     JJ = np.mod(JJ0 + J[ii] - 1, M)
    #     KK = np.mod(KK0 + K[ii] - 1, L)
    #     MM[ii, ...] = np.asarray(data)[II, JJ, KK]

    #     if ii < 3:
    #         tmpdiff = np.diff(data, axis=ii)
    #         for jj in range(ii+1, 3):
    #             MM[l, ...] = np.diff(tmpdiff, axis=jj).mean()
    #             l += 1

    for i in range(N):
        for j in range(M):
            for k in range(L):
                if 0 < i < N-1 and 0 < j < M-1 and 0 < k < L-1:
                    chunk[...] = data[i-1:i+2, j-1:j+2, k-1:k+2]

                else:
                    # Slower version
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                chunk[ii, jj, kk] = data[
                                    mymod(i+ii-1, N),
                                    mymod(j+jj-1, M),
                                    mymod(k+kk-1, L)]

                for l in range(6):
                    MM[i, j, k, l] = chunk[I[l]+1, J[l]+1, K[l]+1]

                l = 0
                for ii in range(3):
                    tmpdiff = np.diff(chunk, axis=ii)
                    for jj in range(ii+1, 3):
                        MM[i, j, k, 6+l] = np.diff(tmpdiff, axis=jj).mean()
                        l += 1

    my_print('Correctly set matrices')
    # Get all coefficients at once
    A = np.dot(np.swapaxes(MM, 0, 3), XXinv)

    # Get AA matrix
    AA = np.zeros((N, M, L, 3, 3), dtype=np.float64)
    rhs = np.zeros((N, M, L, 3), dtype=np.float64)

    for i in range(N):
        for j in range(M):
            for k in range(L):
                AA[i, j, k, 0, 0] = A[i, j, k, 0] * 2
                AA[i, j, k, 0, 1] = A[i, j, k, 6]
                AA[i, j, k, 0, 2] = A[i, j, k, 7]

                AA[i, j, k, 1, 0] = A[i, j, k, 6]
                AA[i, j, k, 1, 1] = A[i, j, k, 1] * 2
                AA[i, j, k, 1, 2] = A[i, j, k, 8]

                AA[i, j, k, 2, 0] = A[i, j, k, 7]
                AA[i, j, k, 2, 1] = A[i, j, k, 8]
                AA[i, j, k, 2, 2] = A[i, j, k, 2] * 2

                rhs[i, j, k, 0] = -A[i, j, k, 3]
                rhs[i, j, k, 1] = -A[i, j, k, 4]
                rhs[i, j, k, 2] = -A[i, j, k, 5]

    AAinv = np.linalg.inv(AA)
    my_print('Inverted matrix of coefficients')

    xyz = np.zeros((N, M, L, 3), dtype=np.float64)
    mask = np.zeros((N, M, L), dtype=np.int16)

    count = 0
    for i in range(N):
        for j in range(M):
            for k in range(L):
                pos = AAinv[i, j, k, :, :] @ rhs[i, j, k, :]

                mask[i, j, k] = (-1 < pos[0] < 1 and
                                 -1 < pos[1] < 1 and
                                 -1 < pos[2] < 1)
                if mask[i, j, k]:
                    for idim in range(3):
                        xyz[i, j, k, idim] = pos[idim]
                    count += 1

    AA_small = np.zeros((count, 3, 3), dtype=np.float64)
    xyz_small = np.zeros((count, 3), dtype=np.float64)

    # Select only the values
    count = 0
    for i in range(N):
        cell_center[0] = i
        for j in range(M):
            cell_center[1] = j
            for k in range(L):
                cell_center[2] = k

                if mask[i, j, k]:
                    AA_small[count, :, :] = AA[i, j, k, :, :]
                    for idim in range(3):
                        xyz_small[count, idim] = xyz[i, j, k, idim] + cell_center[idim]
                    count += 1


    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(AA_small)

    my_print('Computed eigenvalues')
    return xyz_small, eigvals, AA_small

@cython.wraparound(False)
@cython.cdivision(True)
cdef int mymod(int i, int N) nogil:
   if i >= N:
       return i - N
   elif i < 0:
       return i + N
   else:
       return i

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef double[:, :, :] set_data(double[:, :, :] data,
#                    int i0, int j0, int k0,
#                    int N, int M, int L):
#     cdef double newData[3][3][3]
#     cdef double[:, :, :] newDataView = newData
#     cdef int i, j, k, ii, jj, kk
#     cdef double data0

#     data0 = data[i0, j0, k0]

#     for i in range(-1, 2):
#         ii = (i0+i) % N
#         for j in range(-1, 2):
#             jj = (j0+j) % M
#             for k in range(-1, 2):
#                 kk = (k0+k) % L
#                 newData[i][j][k] = data[ii, jj, kk] - data0

#     my_print(np.array(newData))

#     return newDataView
