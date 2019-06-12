from scipy.spatial import cKDTree
from py_extrema.extrema import ExtremaFinder
import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
from unyt import unyt_array

from .extrema import logger
from .utils import measure_hessian, measure_third_derivative, get_xyz_keys


class SloppingSaddle(object):
    """A class to detect slopping saddle point by successive smoothing.

    Parameters
    ----------
    extrema_finder : ExtremaFinder instance
    Rgrid : array like
        Smoothing scales, given in the same unit as the box size.
    """

    def __init__(self, extrema_finder, Rgrid):
        if not issubclass(type(extrema_finder), ExtremaFinder):
            raise Exception('Passed argument is not of type extrema finder.')

        self.ef = extrema_finder

        if not isinstance(Rgrid, unyt_array):
            Rgrid = self.ef.array(Rgrid, 'Mpc')
        self.Rgrid = Rgrid

    def compute_middle(self, x, y):
        boxlen = self.ef.data_shape[0]

        dx = x-y
        center = (x + y) / 2
        center[dx > +boxlen/2] -= boxlen / 2
        center[dx < -boxlen/2] += boxlen / 2

        return center % boxlen

    def detect_extrema(self, iRw=1):
        '''Compute the critical events found in a dataset.

        Parameters
        ----------
        iRw : int, default 1
            The thickness in smoothing scale dimension to look for critical
            points pair. See notes.
        ndim : int, optional, default 3

        Notes
        -----
        The algorithm first computes "heads" (critical points with no successor
        at next smoothing scale). It then tries to find pairs of heads of
        consecutive kind (e.g. peaks with filaments) between R[i] and R[i+iRw].
        '''
        ndim = self.ef.ndim

        # Extract relevant information from dsx
        dimensions = self.ef.data_raw.shape[0]
        smoothing_scales = self.Rgrid.to('pixel').value

        pos_keys = ['x', 'y', 'z'][:ndim]

        all_ext = []
        logger.debug('Building trees')
        for iR, R in enumerate(tqdm(smoothing_scales, desc='Building trees')):
            ext = self.ef.find_extrema(R).as_dataframe()
            ext['iR'] = iR
            ext['R'] = R
            all_ext.append(ext)

        # Build the dataframe containting the extrema
        ds = pd.concat(all_ext)
        ds['uid'] = np.arange(len(ds))
        ds['head'] = True
        ds = ds.reset_index().set_index(['iR', 'kind'])

        logger.debug('Locating heads')
        # Find all points that do not have a successor at a larger
        # smoothing scales ("heads")
        for kind in tqdm(range(ndim+1), leave=False):
            for iR in range(1, len(smoothing_scales)):
                if not ((iR, kind) in ds.index and (iR-1, kind) in ds.index):
                    continue
                p1 = ds.loc[(iR-1, kind), pos_keys].values % dimensions
                p2 = ds.loc[(iR,   kind), pos_keys].values % dimensions

                # Find elements of p2 in p1 (all should match!)
                t = cKDTree(p1, boxsize=dimensions)

                d, iprev = t.query(
                    p2,
                    distance_upper_bound=smoothing_scales[iR])
                ok = np.isfinite(d)
                head = np.ones(p1.shape[0], dtype=bool)
                head[iprev[ok]] = False

                ds.loc[(iR-1, kind), 'head'] = head

        # Build a tree out of the different heads
        heads = ds[ds['head']]

        pairs = []
        skip_uid = set()

        def kind_iter(iR, k0, k1, trees, slice_R):
            # Helper function that computes pairs of head
            h0 = heads.loc[(slice_R, k0), slice(None)]
            h1 = heads.loc[(slice_R, k1), slice(None)]
            if iRw == 1:
                p0 = h0[pos_keys].values
            else:
                p0 = h0[pos_keys + ['R']].values

            if len(h0) == 0 or len(h1) == 0:
                return [], [], []

            t1 = trees[k1]

            d, inext = t1.query(p0, distance_upper_bound=smoothing_scales[iR])
            ok = inext < len(t1.data)
            uid0 = h0[ok].uid.values
            uid1 = h1.iloc[inext[ok]]['uid'].values
            uids = np.sort(np.array((uid0, uid1)), axis=0)
            return uids[0], uids[1], d[ok] / smoothing_scales[iR]

        u0 = []
        u1 = []
        dist = []

        logger.debug('Locating critical events')
        for iR in tqdm(range(len(smoothing_scales)-iRw+1), leave=False):
            trees = {}
            slice_R = slice(iR, iR+iRw)

            # Look once in each direction (+kind, -kind)
            for kind in range(ndim+1):
                if iRw == 1:
                    p = heads.loc[(slice_R, kind), pos_keys].values
                    boxsize = [dimensions]*ndim
                else:
                    p = heads.loc[(slice_R, kind), pos_keys + ['R']].values
                    boxsize = [dimensions]*ndim + [smoothing_scales[-1]*100]
                if len(p) > 0:
                    # Wrap dimensions to prevent negative inputs
                    p = p % dimensions
                    trees[kind] = cKDTree(p, boxsize=boxsize)
            for kind in range(ndim):
                uid0, uid1, d = kind_iter(iR, kind, kind+1, trees, slice_R)
                u0.extend(uid0)
                u1.extend(uid1)
                dist.extend(d)

                uid0, uid1, d = kind_iter(iR, ndim-kind, ndim-kind-1, trees, slice_R)
                u0.extend(uid0)
                u1.extend(uid1)
                dist.extend(d)

        # Sort by increasing distances
        order = np.argsort(dist)
        u0 = np.array(u0)[order]
        u1 = np.array(u1)[order]
        dist = np.array(dist)[order]

        # Select each uid only once
        for uid0, uid1, d in zip(u0, u1, dist):
            if uid0 not in skip_uid or uid1 not in skip_uid and d < 1.5:
                pairs.append((uid0, uid1))
                skip_uid.add(uid0)
                skip_uid.add(uid1)

        # Given the merging pairs, compute the critical events
        uids = np.array(pairs).T
        ds_by_uid = ds.reset_index().set_index('uid')

        h0 = ds_by_uid.loc[uids[0]]
        h1 = ds_by_uid.loc[uids[1]]
        p1 = h0[pos_keys].values
        p2 = h1[pos_keys].values

        logger.debug('Computing critical point properties')
        data = {}
        for col in h0.columns:
            if h0[col].dtype == int:
                data[col] = (h0[col].values + h1[col].values) // 2
            else:
                data[col] = (h0[col].values + h1[col].values) / 2
        data['kind'] = (h0['kind'].values + h1['kind'].values - 1)//2
        L = dimensions
        # Compute position accounting for periodic boundaries
        pp = np.where(p1-p2 > L/2, (p1+p2-L)/2,
                      np.where(p1-p2 < -L/2,
                               (p1+p2+L)/2,
                               (p1+p2)/2))

        # Copy positional data
        for idim, pk in enumerate(pos_keys):
            data[pk] = pp[:, idim]

        critical_events = pd.DataFrame(data)

        self.critical_events = critical_events
        return self.critical_events
