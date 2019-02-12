from scipy.spatial import cKDTree as KDTree
from py_extrema.extrema import ExtremaFinder
import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
from unyt import unyt_array
from collections import namedtuple

from .extrema import logger

ExtrData = namedtuple('ExtremaData', ['tree', 'data'])

class SloppingSaddle(object):
    """A class to detect slopping saddle point by successive smoothing."""

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

    @property
    def trees(self):
        if hasattr(self, '_trees'):
            return self._trees
        ndim = self.ef.ndim
        boxsize = self.ef.data_shape[0]
        Rgrid = self.Rgrid.to('pixel').value
        trees = {
            kind: {
                iR: None
                for iR, _ in enumerate(Rgrid)
            } for kind in range(ndim+1)}

        for iR, R in enumerate(tqdm(Rgrid, desc='Building trees')):
            ext = self.ef.find_extrema(R).as_dataframe()
            for k in range(ndim+1):
                mask = (ext.kind == k)
                pos = ext.loc[mask, ['x', 'y', 'z']].values
                trees[k][iR] = ExtrData(
                    tree=KDTree(np.mod(pos, boxsize),
                                boxsize=boxsize),
                    data=ext.loc[mask])

        self._trees = trees
        return self._trees

    def detect_extrema(self):
        ndim = self.ef.ndim
        trees = self.trees

        # Compute cross tree distances. Slopping saddle are found where
        # the closest point is of different kind.
        ss_points = []
        Rgrid = self.Rgrid.to('pixel').value
        for iR, R in enumerate(tqdm(Rgrid[:-1],
                                    desc='Finding s. saddle')):
            dR = Rgrid[iR+1] - R
            for kind in range(1, ndim):
                # Here we compare the distance from the current critical points
                # to points at the next smoothing scale and next kind
                # of critical points. There are two possibilities:
                # 1. there is one critical point at the next smoothing scale:
                #    the critical point subsists
                # 2. there is one critical point _of another kind_ at
                #    the current scale: the critical point disappears
                t = trees[kind][iR].tree

                # Compute smallest distance to same kind at next
                # smoothing scale
                tnextR = trees[kind][iR + 1].tree
                dnextR, inextR = tnextR.query(
                    t.data, distance_upper_bound=2*R)

                # Compute distance to other critical points
                # previous kind
                tprev = trees[kind - 1][iR].tree
                dprev, iprev = tprev.query(t.data,
                                           distance_upper_bound=2*R)

                # next kind
                tnext = trees[kind + 1][iR].tree
                dnext, inext = tnext.query(t.data,
                                           distance_upper_bound=2*R)

                # Check :
                # * there is no crit. pt. of same kind within a few dR
                # * there is a crit. pt. at same scale of next kind
                #   (e.g. saddle point-peak)
                mask_prev = (dprev < dnextR) & (dprev < dnext) \
                  & np.isfinite(dprev)
                mask_next = (dnext < dnextR) & (dnext < dprev) \
                  & np.isfinite(dnext)

                mask_both = mask_prev | mask_next

                logger.debug(
                    'Slopping saddle rate %s: %.2f%%' %
                    (kind, mask_both.sum() / mask_both.shape[0] * 100))

                # Compute position -- prev
                new_ss_pos = self.compute_middle(
                    t.data[mask_prev],
                    tprev.data[iprev[mask_prev]])

                # Compute data
                keys = ['l1', 'l2', 'l3', 'h11', 'h22', 'h33', 'h12', 'h13', 'h23', 'dens']
                A = trees[kind][iR].data.loc[mask_prev][keys]
                B = trees[kind-1][iR].data.iloc[iprev[mask_prev]][keys]

                datacur = A.values
                dataprev = B.values

                new_data = (datacur + dataprev) / 2

                for ii, pos in enumerate(new_ss_pos):
                    ss_points.append((kind-1, iR+1, R, *new_data[ii, :], *pos))

                # Compute position -- next
                new_ss_pos = self.compute_middle(
                    t.data[mask_next],
                    tnext.data[inext[mask_next]])

                # Compute data
                A = trees[kind][iR].data.loc[mask_next][keys]
                B = trees[kind + 1][iR].data.iloc[inext[mask_next]][keys]

                datacur = A.values
                dataprev = B.values

                new_data = (datacur + dataprev) / 2

                for ii, pos in enumerate(new_ss_pos):
                    ss_points.append((kind, iR+1, R, *new_data[ii, :], *pos))

        names = ['kind', 'iR', 'R'] + keys
        for e in 'xyz'[:ndim]:
            names.append(e)

        self.slopping_saddle = pd.DataFrame(ss_points,
                                            columns=names)
        return self.slopping_saddle
