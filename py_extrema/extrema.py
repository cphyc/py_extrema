import numpy as np
from pyfftw.interfaces import numpy_fft as fft

from unyt import UnitRegistry
import unyt as U
from unyt import unyt_array, unyt_quantity
from unyt.dimensions import length

from numba import jit, njit
import logging
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import interpn
import numexpr as ne

from .utils import FiniteDictionary, CriticalPoints, unravel_index, solve

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


@jit
def copy_xyz(xyz_rela):
    '''Keep only the points found within 1pixel from their parent cell'''
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


@njit
def distance2(a, b, N):
    """Compute the distance between two vectors with wrapping of size N."""
    d2 = 0
    for i in range(a.shape[0]):
        aa = a[i]
        bb = b[i]
        dx = aa - bb
        if dx < -N/2:
            d2 += (dx + N)**2
        elif dx > N/2:
            d2 += (dx - N)**2
        else:
            d2 += (dx)**2
    return d2


@njit
def _cleanup_pairs_KDTree(xyz, xc, kind, pairs, N, shape, grad_norm=None):
    skip = np.zeros(len(xyz), dtype=np.uint8)
    for ip in range(len(pairs)):
        i, j = pairs[ip]
        if not skip[i] and not skip[j] and kind[i] == kind[j]:
            # Keep point closest to center of cell
            _xc = xc[i]
            if grad_norm is None:
                di = np.sum((xyz[i] - _xc)**2)
                dj = np.sum((xyz[j] - _xc)**2)
            else:
                di = grad_norm[i]
                dj = grad_norm[j]
            if di > dj:
                skip[i] = True
            else:
                skip[j] = True

    return skip


def cleanup_pairs_KDTree(xyz, kind, data_shape, dmin, grad):
    npoint, ndim = xyz.shape
    N = data_shape[0]
    logger.debug('Building KDTree')
    # TODO: support non square domains
    if not np.all(np.asarray(data_shape) == data_shape[0]):
        raise Exception('All axis should have the same dimension.')
    if len(xyz) == 0:
        return np.ones(0, dtype=bool)
    tree = KDTree(xyz, boxsize=data_shape[0], copy_data=True)
    pairs = tree.query_pairs(dmin, p=np.inf, output_type='ndarray')
    logger.debug('Removing close pairs')
    xc = np.round(xyz + 0.5) - 0.5
    skip = _cleanup_pairs_KDTree(xyz, xc, kind, pairs, N, data_shape,
                                 np.linalg.norm(grad, axis=1)).astype(bool)
    return ~skip


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


class ExtremaFinder(object):
    """A class to smooth an extract extrema from a n-dimensional field.

    Parameters
    ----------
    data : n-dimensional array
        The input field as a ndim array.
    boxlen : float
        Size of the box. All scales are then given in the same unit as boxlen.
    cache_len : int, optional
        Size of the cache containing the smoothed field. Defaults to 2.
    clean_pairs_method : str, optional
        The method used to discard close extrema. Defaults to 'KDTree'.
        See notes.
    nthreads : int
        Number of threads to use for FFT, etc.
    logleve : int
        Set to a low value for more verbosity

    Notes
    -----
    When discarding extrema, there are three methods available:
    * 'none': do not discard close extrema
    * 'direct': use an approach based on the relative distance between extrema.
                This scales as N^2 where N is the number of extrema.
    * 'KDTree': same as previous but using a KDTree, resulting in better perfs
                for large number of extrema.
    The default option gives the best result for large number of extrema.
    If the number of extrema is small, switch back to the 'direct' method.
    """

    ndim = None    # Number of dimensions
    boxlen = None  # Size of the box in physical units
    data_smooth = None    # Cache containing the real-space smoothed fields
    data_smooth_f = None  # Cache containing the Fourier-space smoothed fields
    extrema = None        # Dictionary containing the extrema
    data_raw = None    # Real-space raw data
    data_raw_f = None  # Fourier-space raw data

    kgrid = None  # The grid of Fourier kx, ky, ... modes
    k2 = None     # The grid of Fourier k^2

    clean_pairs_methods = ['KDTree', 'none', 'direct']
    clean_pairs_method = None

    dmin = 1  # Minimum distance between 2 points to merge them

    FFT_args = None

    def __init__(self, data, cache_len=2, clean_pairs_method='KDTree',
                 nthreads=1, loglevel=None, boxlen=1):

        self.registry = reg = UnitRegistry()
        reg.add("pixel", base_value=float((U.Mpc * boxlen / data.shape[0]).to('m')),
                dimensions=length)

        if loglevel:
            logger.setLevel(loglevel)

        self.ndim = data.ndim
        self.boxlen = boxlen
        ndim = self.ndim
        self.nthreads = nthreads
        self.clean_pairs_method = clean_pairs_method

        self.FFT_args = dict(threads=nthreads)

        self.data_raw = data
        self.data_shape = data.shape

        # Initialize caches
        self.data_smooth = FiniteDictionary(cache_len)
        self.data_smooth_f = FiniteDictionary(cache_len)
        self.extrema = {}

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

    def array(self, value, units):
        return unyt_array(value, units, registry=self.registry)

    def quantity(self, value, units):
        return unyt_quantity(value, units, registry=self.registry)

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
        if isinstance(R, unyt_quantity):
            R = float(R.to('pixel'))
        if R in self.data_smooth:
            return self.data_smooth[R]

        logger.debug('Smoothing at scale %.3f', R)
        data_f = self.data_raw_f * np.exp(-self.k2 * R**2 / 2)
        self.data_smooth_f[R] = data_f
        self.data_smooth[R] = fft.irfftn(data_f, **self.FFT_args)
        return self.data_smooth[R]

    def clean_pairs(self, xyz0, kind0, grad):
        if self.clean_pairs_method == 'KDTree':
            boxlen = self.data_raw.shape
            mask = cleanup_pairs_KDTree(
                np.mod(xyz0, boxlen),
                kind0, self.data_raw.shape, self.dmin, grad)

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
            k1 = kgrid[idim]
            self.grad_f[idim, ...] = ne.evaluate('data_f * 1j * k1')
            for idim2 in range(idim, ndim):
                k2 = kgrid[idim2]
                grad_f = self.grad_f[idim, ...]
                self.hess_f[ihess, ...] = ne.evaluate(
                    'grad_f * 1j * k2')

                indices[idim, idim2] = indices[idim2, idim] = ihess
                ihess += 1

        logger.debug('Inverse FFT of gradient')
        # Get them back in real space
        self.grad[...] = fft.irfftn(self.grad_f, axes=range(1, ndim+1), **self.FFT_args)
        logger.debug('Inverse FFT of hessian')
        self.hess[...] = fft.irfftn(self.hess_f, axes=range(1, ndim+1), **self.FFT_args)
        logger.debug('Curvature')
        if ndim == 3:
            a = self.hess[indices[0, 0]]
            b = self.hess[indices[1, 1]]
            c = self.hess[indices[2, 2]]
            d = self.hess[indices[0, 1]]
            e = self.hess[indices[0, 2]]
            f = self.hess[indices[1, 2]]
            self.curvature[...] = ne.evaluate(
                'a*b*c - c*d**2 - b*e**2 + 2*d*e*f - a*f**2',
                local_dict=dict(a=a, b=b, c=c, d=d, e=e, f=f))
        elif ndim == 2:
            a = self.hess[indices[0, 0]]
            b = self.hess[indices[1, 1]]
            c = self.hess[indices[0, 1]]
            self.curvature[...] = ne.evaluate(
                'a*b - c**2',
                local_dict=dict(a=a, b=b, c=c))
        else:
            self.curvature[...] = np.linalg.det(self.hess[indices, ...].T.copy()).T

        return indices

    def find_extrema(self, R):
        """Find the extrema of the field smoothed at scale R (in pixel unit).

        Return
        ------
        data: CriticalPoints
              The set of critical points found.
        """
        if isinstance(R, unyt_quantity):
            R = float(R.to('pixel'))
        if R in self.extrema:
            return self.extrema[R]
        ndim = self.ndim
        shape = self.data_shape

        if R not in self.data_smooth_f:
            self.smooth(R)

        indices = self.compute_derivatives(R)
        grid = np.linspace(0, shape[0], shape[0])

        rhs = -self.grad
        lhs = self.hess[indices.flatten(), ...].reshape(ndim, ndim, *shape)

        logger.debug('Solving linear system to find extrema')
        # Find the location of the points (in relative coordinates)
        # shape (N, N, N, ndim)
        xyz_rel = solve(np.asfortranarray(lhs.T),
                        np.asfortranarray(rhs.T))

        logger.debug('Discarding far extrema')
        # shapes (npoint, ndim) & (npoint, )
        xyz0, mask0 = copy_xyz(xyz_rel)
        mask0 = (mask0 == 1)

        # shape (npoint)
        dens0 = interpn([grid]*ndim, self.smooth(R), xyz0 % shape)

        # shape (npoint, ndim, ndim)
        # For the interpolation, move the hessian component at the end
        hess0 = (interpn([grid]*ndim, np.moveaxis(self.hess, 0, -1), xyz0 % shape)
                 [..., indices]
                 .reshape(-1, ndim, ndim))
        grad0 = (interpn([grid]*ndim, np.moveaxis(self.grad, 0, -1), xyz0 % shape)
                 .reshape(-1, ndim))

        logger.debug('Computing eigenvalues')
        # shape (npoint, ndim)
        eigvals0 = np.linalg.eigvalsh(hess0)

        # shape (npoint, )
        kind0 = (eigvals0 > 0).sum(axis=1)

        logger.info('Found %s extrema.', len(kind0))
        logger.debug('Cleaning up pairs with method %s',
                     self.clean_pairs_method)

        # Remove duplicate points
        mask = self.clean_pairs(xyz0, kind0, grad0)

        # Add units
        pos = self.array(xyz0[mask], 'pixel')
        hess = self.array(hess0[mask], '1/pixel**2')
        eigvals = self.array(eigvals0[mask], '1/pixel**2')
        dens = dens0[mask]
        sigma = self.smooth(R).std() * np.ones_like(dens)

        # Convert to physical units
        data = CriticalPoints(
            pos=pos,
            eigvals=eigvals,
            kind=kind0[mask],
            hessian=hess,
            npt=mask.sum(),
            dens=dens,
            sigma=sigma
        )

        self.extrema[R] = data
        return data
