from py_extrema.extrema import ExtremaFinder
import numpy as np

N = 100
ndim = 2
np.random.seed(16091992)
data = {}
for ndim in [2]:
    data[ndim] = np.random.rand(*[N]*ndim)

_PLOT = True

if _PLOT:
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt


def testSmoothing():
    ef = ExtremaFinder(data[2])
    data_smoothed = ef.smooth(N/10)

    # TODO: check we have correctly smoothed at scale R
    # There may be some missing value (e.g. 1/2pi)
    if _PLOT:
        plt.imshow(data_smoothed)

        plt.title(rf'ndim={ndim}')
        plt.savefig('smoothed.pdf')


def testExtrema1D():
    ef = ExtremaFinder(data[1])
    R = N/5
    extr = ef.find_extrema(R)

    field = ef.smooth(R)

    if _PLOT:
        plt.cla()
        x = np.arange(N)
        interp = interp1d(x, field)
        plt.plot(field)
        for kind in [0, 1]:
            m = extr.kind == kind
            plt.plot(extr.pos[m], interp(extr.pos[m]), 'o', label=f'kind={kind}')

        plt.legend()
        plt.savefig('extrema_1D.pdf')
        plt.clf()


def testExtrema2D():
    ef = ExtremaFinder(data[2])
    R = N/5
    extr = ef.find_extrema(R)

    field = ef.smooth(R)

    if _PLOT:
        plt.cla()
        plt.imshow(field.T)
        for kind in [0, 1, 2]:
            m = extr.kind == kind
            plt.plot(extr.pos[m, 0], extr.pos[m, 1], '.', label=f'kind={kind}')

        plt.xlim(0, N)
        plt.ylim(0, N)
        plt.legend()
        plt.savefig('extrema_2D.pdf')
        plt.clf()
