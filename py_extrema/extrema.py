import numpy as np
from collections import OrderedDict
try:
    from pyfftw.interfaces import numpy_fft as fft
except ImportError:
    print('Could not load pyfftw, falling back to numpy.')
    fft = np.fft


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


class ExtremaFinder(object):
    smooth_cache = None    # Cache containing the real-space smoothed fields
    smooth_f_cache = None  # Cache containing the Fourier-space smoothed fields
    data_raw = None    # Real-space raw data
    data_raw_f = None  # Fourier-space raw data

    def __init__(self, data, cache_len=10):
        self.data_raw = data
        self.data_shape = data.shape

        # Initialize caches
        self.smooth_cache = FiniteDictionary(cache_len)
        self.smooth_f_cache = FiniteDictionary(cache_len)

        self.build_kgrid()
        self.fft_forward()

    def build_kgrid(self):
        shape = self.data_shape
        kall = ([np.fft.fftfreq(_) for _ in shape[:-1]] +
                [np.fft.rfftfreq(_) for _ in shape[-1:]])
        self.kall = kall
        self.k2 = (np.asarray(np.meshgrid(*kall, indexing='ij'))**2).sum(axis=0)

    def fft_forward(self):
        data = self.data_raw
        self.data_raw_f = fft.rfftn(data)

    def smooth(self, R):
        if R in self.smooth_cache:
            return self.smooth_cache[R]

        data_f = self.data_raw_f * np.exp(-self.k2 * R**2)
        self.smooth_f_cache[R] = data_f
        self.smooth_cache[R] = fft.irfftn(data_f)
        return self.smooth_cache[R]

    # def find_extrema(self, R):
