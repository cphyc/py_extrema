from numba import jit
from collections import namedtuple, OrderedDict
import numpy as np


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


CriticalPoints = namedtuple('CriticalPoints',
                            ['pos', 'eigvals', 'kind', 'hessian', 'npt'])


@jit(nopython=True)
def unravel_index(index, shape):
    n = len(shape)
    result = np.zeros(n, dtype=np.int32)
    for i in range(n-1, -1, -1):
        s = shape[i]
        result[i] = index % s
        index //= s

    return result
