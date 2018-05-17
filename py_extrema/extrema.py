import numpy as np
from pyfftw.interfaces import numpy_fft as fft

from numba import jit
import logging
from scipy.spatial import cKDTree as KDTree
from .utils import FiniteDictionary, CriticalPoints, unravel_index

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a stream handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Add handler to logger
logger.addHandler(handler)


@jit(nopython=True)
def copy_xyz(xyz_rela):
    xyz_rel = np.ascontiguousarray(xyz_rela.T)
    shape = xyz_rel.shape[1:]

    ndim = xyz_rel.shape[0]
    xyz_copy = xyz_rel.reshape(ndim, -1)

    N = xyz_copy.shape[1]

    count = 0
    for ii in range(N):
        ok = True
        for idim in range(ndim):
            ok &= -1 < xyz_copy[idim, ii] < 1

        if ok:
            count += 1

    xyz = np.zeros((ndim, count), dtype=xyz_rel.dtype)
    mask = np.zeros(N, dtype=np.int8)

    count = 0
    for ii in range(N):
        ok = True
        for idim in range(ndim):
            ok &= -1 < xyz_copy[idim, ii] < 1
        mask[ii] = ok
        if ok:
            ijk = list(unravel_index(ii, shape))
            for idim in range(ndim):
                xyz[idim, count] = ijk[idim]
            xyz[:, count] += xyz_copy[:, ii]
            count += 1

    return xyz.T, mask

@jit
def distance2(a, b, N):
    """Compute the distance between two vectors with wrapping of size N."""
    d2 = 0
    for i in range(a.shape[0]):
        aa = a[i]
        bb = b[i]
        if aa - bb < -N/2:
            d2 += (aa - bb + N)**2
        elif aa - bb > N/2:
            d2 += (aa - bb - N)**2
        else:
            d2 += (aa - bb)**2
    return d2


@jit(nopython=True)
def _cleanup_pairs_KDTree(xyz, kind, pairs, N, shape):
    skip = np.zeros_like(kind, dtype=np.int8)
    for i, j in pairs:
        if kind[i] == kind[j] and skip[i] != 1 and skip[j] != 1:
            ijk1 = unravel_index(i, shape)
            ijk2 = unravel_index(j, shape)

            d1 = distance2(xyz[i], ijk1, N)
            d2 = distance2(xyz[j], ijk2, N)

            # Keep point closest to cell center
            if d1 > d2:
                xyz[i, :] = -1
                skip[i] = 1
            else:
                xyz[j, :] = -1
                skip[j] = 1


def cleanup_pairs_KDTree(xyz, kind, data_shape, dmin):
    npoint, ndim = xyz.shape
    N = data_shape[0]
    logger.debug('Building KDTree')
    # TODO: support non square domains
    if not np.all(np.asarray(data_shape) == data_shape[0]):
        raise Exception('All axis should have the same dimension.')
    tree = KDTree(xyz, boxsize=data_shape[0])

    pairs = list(tree.query_pairs(dmin))
    logger.debug('Removing close pairs')
    _cleanup_pairs_KDTree(xyz, kind, pairs, N, data_shape)


@jit(nopython=True)
def cleanup_pairs_N2(xyz, kind, data_shape):
    N = data_shape[0]
    npoint, ndim = xyz.shape

    for i in range(npoint):
        p1 = xyz[i, :]
        k1 = kind[i]
        ijk1 = unravel_index(i, data_shape)
        d1 = distance2(p1, ijk1, N)
        if np.all(p1 == -1):
            continue
        for j in range(i+1, npoint):
            p2 = xyz[j, :]
            if k1 != kind[j] or np.all(p2 == -1):
                continue

            if distance2(p1, p2, N) < 1:
                ijk2 = unravel_index(j, data_shape)
                d2 = distance2(p2, ijk2, N)

                # Keep 2
                if d1 > d2:
                    xyz[i, :] = -1
                else:
                    xyz[j, :] = -1


#@jit(nopython=True)
# def cleanup_pairsold(xyz, kind, data_shape):
#     npoint, ndim = xyz.shape
#     tmp = np.zeros(data_shape, dtype=np.int32)
#     N = data_shape[0]

#     ijk = [(0,)]*ndim
#     iold = 0

#     Nextr = xyz.shape[0]
#     for inew in range(Nextr):
#         pos = xyz[inew, :]
#         for idim in range(ndim):
#             ijk[idim] = (int((np.round(xyz[inew, idim]))) % N, )

#         # Get index of old value
#         iold = tmp[ijk][0]
#         if iold > 0:
#             oldpos = xyz[iold, ...]
#             # Old is closer than old, keep new
#             if ( distance2(oldpos, ijk, N) > distance2(pos, ijk, N) and
#                  kind[inew] == kind[iold]):
#                 tmp[ijk] = inew
#         else:
#             tmp[ijk] = inew
#     indexes = tmp[tmp > 0]

#     return xyz[indexes, :], indexes


class ExtremaFinder(object):
    """A class to smooth an extract extrema from a n-dimensional field."""

    ndim = None  # Number of dimensions
    data_smooth = None    # Cache containing the real-space smoothed fields
    data_smooth_f = None  # Cache containing the Fourier-space smoothed fields
    data_raw = None    # Real-space raw data
    data_raw_f = None  # Fourier-space raw data

    kgrid = None  # The grid of Fourier kx, ky, ... modes
    k2 = None     # The grid of Fourier k^2

    clean_pairs_methods = ['KDTree', 'none', 'direct']
    clean_pairs_method = None

    dmin = 1  # Minimum distance between 2 points to merge them

    FFT_args = None

    def __init__(self, data, cache_len=10, clean_pairs_method='KDTree',
                 nthreads=1, loglevel=None):

        if loglevel:
            logger.setLevel(loglevel)

        self.ndim = data.ndim
        ndim = self.ndim
        self.nthreads = nthreads
        self.clean_pairs_method = clean_pairs_method

        self.FFT_args = dict(threads=nthreads)

        self.data_raw = data
        self.data_shape = data.shape

        # Initialize caches
        self.data_smooth = FiniteDictionary(cache_len)
        self.data_smooth_f = FiniteDictionary(cache_len)

        # Perform first FFT + prepare k grid
        self.build_kgrid()
        self.fft_forward()

        # Initialize placeholders
        shape = list(self.data_raw.shape)
        shape_f = list(self.data_raw_f.shape)
        self.grad = np.zeros([ndim] + shape,
                             dtype=self.data_raw.dtype)
        self.hess = np.zeros([ndim*(ndim+1)//2] + shape,
                             dtype=self.data_raw.dtype)
        self.curvature = np.zeros(shape, dtype=self.data_raw.dtype)

        self.grad_f = np.zeros([ndim] + shape_f,
                               dtype=self.data_raw_f.dtype)
        self.hess_f = np.zeros([ndim*(ndim+1)//2] + shape_f,
                               dtype=self.data_raw_f.dtype)

    def build_kgrid(self):
        """Build the grid of k coefficients.

        Note
        ----
        This is made tricky by the FFT implementation. Usually, the
        last dimension is roughly half the input size (and complex),
        so the code has to take that into account.

        The last dimension is compressed because for real-input
        fields, all the information is stored in the complex part
        (twice the info of a real), so the shape is half the input
        size.
        """
        twopi = 2*np.pi
        shape = self.data_shape
        kall = ([np.fft.fftfreq(_)*twopi for _ in shape[:-1]] +
                [np.fft.rfftfreq(_)*twopi for _ in shape[-1:]])
        self.kgrid = np.asarray(np.meshgrid(*kall, indexing='ij'))
        self.k2 = (self.kgrid**2).sum(axis=0)

    def fft_forward(self):
        """Compute the direct Fourier transform of the input data."""
        logger.debug('Computing FFT of input with %s threads', self.nthreads)
        data = self.data_raw

        # Note: we don't use the defaults argument for the FFT as we
        # don't want any planning effort going to this one
        self.data_raw_f = fft.rfftn(data, threads=self.nthreads)

    def smooth(self, R):
        """Smooth the data at scale R (in pixel unit)."""
        if R in self.data_smooth:
            return self.data_smooth[R]

        logger.debug('Smoothing at scale %.3f', R)
        data_f = self.data_raw_f * np.exp(-self.k2 * R**2 / 2)
        self.data_smooth_f[R] = data_f
        self.data_smooth[R] = fft.irfftn(data_f, **self.FFT_args)
        return self.data_smooth[R]

    def clean_pairs(self, xyz0, kind0):
        if self.clean_pairs_method == 'KDTree':
            cleanup_pairs_KDTree(np.mod(xyz0, self.data_raw.shape),
                                 kind0, self.data_raw.shape, self.dmin)
            mask = np.any(xyz0 != -1, axis=1)

        elif self.clean_pairs_method == 'direct':
            cleanup_pairs_N2(xyz0, kind0, self.data_raw.shape)
            mask = np.any(xyz0 != -1, axis=1)

        elif self.clean_pairs_method == 'none':
            mask = np.ones(len(xyz0), dtype=bool)

        else:
            raise NotImplementedError(
                'The method %s is not implemented to clean '
                'pairs. Available methods are %s ' %
                (self.clean_pairs_method, self.clean_pairs_methods))

        return mask

    def compute_derivatives(self, R):
        """Compute the 1st and 2nd derivatives of the (Fourier) input.

        Argument
        --------
        R: float
            The smoothing scale.

        Return
        ------
        indices: (ndim, ndim, M)
            The M indices of the hessian. M = ndim*(ndim+1)/2.
        """
        if R not in self.data_smooth_f:
            self.smooth(R)

        data_f = self.data_smooth_f[R]
        ndim = self.ndim
        kgrid = self.kgrid
        ihess = 0
        indices = np.zeros((ndim, ndim), dtype=int)

        logger.debug('Computing hessian and gradient in Fourier space.')
        # Compute hessian and gradient in Fourier space
        for idim in range(ndim):
            self.grad_f[idim, ...] = data_f * (1j) * kgrid[idim]
            for idim2 in range(idim, ndim):
                self.hess_f[ihess, ...] = (
                    self.grad_f[idim, ...] * (1j) * kgrid[idim2])

                indices[idim, idim2] = indices[idim2, idim] = ihess
                ihess += 1

        logger.debug('Inverse FFT of gradient and hessian')
        # Get them back in real space
        self.grad[...] = fft.irfftn(self.grad_f, axes=range(1, ndim+1), **self.FFT_args)
        self.hess[...] = fft.irfftn(self.hess_f, axes=range(1, ndim+1), **self.FFT_args)
        self.curvature[...] = np.linalg.det(self.hess[indices, ...].T).T

        return indices

    def find_extrema(self, R):
        """Find the extrema of the field smoothed at scale R (in pixel unit).

        Return
        ------
        data: CriticalPoints
              The set of critical points found.
        """
        ndim = self.ndim
        shape = self.data_shape

        if R not in self.data_smooth_f:
            self.smooth(R)

        indices = self.compute_derivatives(R)

        rhs = -self.grad
        lhs = self.hess[indices.flatten(), ...].reshape(ndim, ndim, *shape)

        logger.debug('Solving linear system to find extrema')
        # Find the location of the points (in relative coordinates)
        # shape (N, N, N, ndim)
        xyz_rel = np.linalg.solve(lhs.T, rhs.T)

        logger.debug('Discarding far extrema')
        # shapes (npoint, ndim) & (npoint, )
        xyz0, mask0 = copy_xyz(xyz_rel)
        mask0 = (mask0 == 1)

        # shape (npoint, ndim, ndim)
        hess0 = (self.hess.reshape(ndim*(ndim+1)//2, -1)
                 [:, mask0]
                 [indices.flatten()]
                 .reshape(ndim, ndim, -1)).T

        logger.debug('Computing eigenvalues')
        # shape (npoint, ndim)
        eigvals0 = np.linalg.eigvalsh(hess0)

        # shape (npoint, )
        kind0 = (eigvals0 > 0).sum(axis=1)

        logger.info('Found %s extrema.', len(kind0))
        logger.debug('Cleaning up pairs with method %s',
                     self.clean_pairs_method)

        # Remove duplicate points
        mask = self.clean_pairs(xyz0, kind0)

        xyz = xyz0[mask]

        data = CriticalPoints(
            pos=xyz, eigvals=eigvals0[mask, ...],
            kind=kind0[mask], hessian=hess0[mask, ...],
            npt=mask.sum()
        )

        return data
