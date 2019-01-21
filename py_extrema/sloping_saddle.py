from scipy.spatial import cKDTree as KDTree
from py_extrema.extrema import ExtremaFinder
import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
from .extrema import logger


class SlopingSaddle(object):
    """A class to detect sloping saddle point by successive smoothing."""

    def __init__(self, extrema_finder, Rgrid):
        if not issubclass(type(extrema_finder), ExtremaFinder):
            raise Exception('Passed argument is not of type extrema finder.')

        self.ef = extrema_finder
        self.Rgrid = Rgrid

    def compute_middle(self, x, y):
        boxlen = self.ef.data_shape[0]

        m1 = x - y > boxlen / 2
        m2 = x - y < -boxlen / 2

        return np.mod(
            np.where(m1, x + y - boxlen,
                     np.where(m2, x + y + boxlen, x + y)) / 2,
            boxlen)

    @property
    def trees(self):
        if hasattr(self, '_trees'):
            return self._trees
        ndim = self.ef.ndim
        boxsize = self.ef.data_shape[0]
        trees = {
            kind: {
                iR: None
                for iR, _ in enumerate(self.Rgrid)
            } for kind in range(ndim+1)}

        for iR, R in enumerate(tqdm(self.Rgrid, desc='Building trees')):
            ext = self.ef.find_extrema(R)
            for k in range(ndim+1):
                mask = (ext.kind == k)
                trees[k][iR] = KDTree(np.mod(ext.pos[mask], boxsize),
                                      boxsize=boxsize)

        self._trees = trees
        return self._trees

    def detect_extrema(self):
        ndim = self.ef.ndim
        trees = self.trees

        # Compute cross tree distances. Sloping saddle are found where
        # the closest point is of different kind.
        ss_points = []
        for iR, R in enumerate(tqdm(self.Rgrid[:-1],
                                    desc='Finding s. saddle')):
            dR = self.Rgrid[iR+1] - R
            for kind in range(ndim):
                # Here we compare the distance from the current critical points
                # to points at the next smoothing scale and next kind
                # of critical points. There are two possibilities:
                # 1. there is one critical point at the next smoothing scale:
                #    the critical point subsists
                # 2. there is one critical point _of another kind_ at
                #    the current scale: the critical point disappears
                t1 = trees[kind][iR]

                # Compute smallest distance
                tnext = trees[kind][iR + 1]
                dnext, inext = tnext.query(t1.data,
                                           distance_upper_bound=2*dR)

                tother = trees[kind+1][iR]
                dother, iother = tother.query(t1.data,
                                              distance_upper_bound=2*R)

                # Check :
                # * there is no crit. pt. of same kind within a few dR
                # * there is a crit. pt. at same scale of next kind
                #   (e.g. saddle point-peak)
                mask = (dother < dnext) & np.isinf(dnext)

                logger.debug('Slopping saddle rate %s: %.2f%%' %
                             (kind, mask.sum() / mask.shape[0] * 100))

                # Compute position
                new_ss_pos = self.compute_middle(t1.data[mask],
                                                 tother.data[iother[mask]])
                for nssp in new_ss_pos:
                    ss_points.append((int(kind), iR+1, R, *nssp))

        names = ['kind', 'iR', 'R']
        for e in 'xyz'[:ndim]:
            names.append(e)

        self.slopping_saddle = pd.DataFrame(ss_points,
                                            columns=names)
        return self.slopping_saddle
