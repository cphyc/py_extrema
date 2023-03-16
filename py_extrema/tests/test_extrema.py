import numpy as np
import pytest

from py_extrema.extrema import ExtremaFinder

N = 100


@pytest.fixture
def data3d():
    np.random.seed(16091992)
    return np.random.normal(size=(N, N, N))


@pytest.fixture
def data2d():
    np.random.seed(16091992)
    return np.random.normal(size=(N, N))


@pytest.fixture
def data1d():
    np.random.seed(16091992)
    return np.random.normal(size=(N))


_PLOT = True

if _PLOT:
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d


def testGradientHessian(data1d):
    ndim = 1
    ef = ExtremaFinder(data1d)
    ef.compute_derivatives(N / 10)

    grad = ef.grad[0]
    hess = ef.hess[0]

    # Finite differences
    fd_grad = np.gradient(ef.smooth(N / 10))
    fd_hess = np.gradient(fd_grad)

    # Check they are close.
    # Note: we drop first item because of boundary conditions
    assert np.allclose(grad[1:-1], fd_grad[1:-1], atol=np.abs(fd_grad).max() / 100)
    assert np.allclose(hess[2:-2], fd_hess[2:-2], atol=np.abs(fd_hess).max() / 100)

    # TODO: check we have correctly smoothed at scale R
    # There may be some missing value (e.g. 1/2pi)
    if _PLOT:
        fig, axes = plt.subplots(nrows=2, sharex=True, squeeze=True)
        axes[0].plot(grad, label="FFT")
        axes[0].plot(fd_grad, ls="--", label="Finite difference")
        axes[0].set_title(rf"Gradient ndim={ndim}")
        axes[0].legend()

        axes[1].plot(hess, label="FFT")
        axes[1].plot(fd_hess, ls="--", label="Finite difference")

        axes[1].set_title(rf"Hessian ndim={ndim}")
        fig.savefig("smoothed_1D.pdf")


def testSmoothing1D():
    N = 100
    data = np.zeros(N)
    x = np.arange(N)
    data[N // 2] = 1

    ef = ExtremaFinder(data)

    # Don't increase that too much as the gaussian may "leak" when
    # wrapping around box
    sigma = N / 20

    data_smoothed = ef.smooth(sigma)
    data_exact = (
        1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - N // 2) ** 2) / (2 * sigma**2))
    )

    assert np.allclose(data_smoothed, data_exact)

    if _PLOT:
        plt.figure()
        plt.plot(data_smoothed, label="FFT")
        plt.plot(
            1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - N // 2) ** 2) / (2 * sigma**2)),
            ls="--",
            label="Exact",
        )
        plt.legend()
        plt.savefig("gaussian.pdf")
        plt.close()


def testSmoothing2D(data2d):
    ndim = 2
    ef = ExtremaFinder(data2d)
    data_smoothed = ef.smooth(N / 10)

    # TODO: check we have correctly smoothed at scale R
    # There may be some missing value (e.g. 1/2pi)
    if _PLOT:
        plt.figure()
        plt.imshow(data_smoothed)

        plt.title(rf"ndim={ndim}")
        plt.savefig("smoothed_2D.pdf")
        plt.close()


def testExtrema1D(data1d):
    ef = ExtremaFinder(data1d)
    R = N / 50
    extr = ef.find_extrema(R)

    field = ef.smooth(R)

    if _PLOT:
        plt.figure()
        x = np.arange(N)
        interp = interp1d(x, field)
        plt.plot(field)
        for kind in [0, 1]:
            m = extr.kind == kind
            plt.plot(extr.pos[m], interp(extr.pos[m]), "o", label=f"kind={kind}")

        plt.legend()
        plt.savefig("extrema_1D.pdf")
        plt.close()


def testExtrema2D(data2d):
    ef = ExtremaFinder(data2d)
    R = N / 20
    extr = ef.find_extrema(R)

    field = ef.smooth(R)

    if _PLOT:
        plt.figure()
        vmax = np.abs(field).max()
        plt.imshow(field.T, vmin=-vmax, vmax=vmax, cmap="bwr")
        for kind in (0, 1, 2):
            m = extr.kind == kind
            plt.plot(extr.pos[m, 0], extr.pos[m, 1], ".", label=f"kind={kind}")

        plt.xlim(0, N)
        plt.ylim(0, N)
        plt.legend()
        plt.savefig("extrema_2D.pdf")
        plt.close()


def testExtrema3D(data3d):
    ef = ExtremaFinder(data3d)
    R = N / 20
    extr = ef.find_extrema(R)

    field = ef.smooth(R)

    if _PLOT:
        plt.figure()
        vmax = np.abs(field).max()
        # Find y coordinate of maximum
        jmax = np.unravel_index(np.argmax(field), field.shape)[1]

        plt.imshow(field.T[:, jmax, :], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        for kind in (0, 1, 2, 3):
            m = (extr.kind == kind) & (np.round(extr.pos[:, 1]) == jmax)
            plt.plot(extr.pos[m, 0], extr.pos[m, 2], ".", label=f"kind={kind}")

        plt.xlim(0, N)
        plt.ylim(0, N)
        plt.legend()
        plt.savefig("extrema_3D.pdf")
        plt.close()
