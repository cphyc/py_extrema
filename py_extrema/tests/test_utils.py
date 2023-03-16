import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import RegularGridInterpolator

from py_extrema.utils import (
    FiniteDictionary,
    gradient,
    measure_gradient,
    measure_hessian,
    solve,
    trilinear_interpolation,
    unravel_index,
)


def testFiniteDictionary():
    fd = FiniteDictionary(10)

    for i in range(10):
        fd[i] = i

    for i in range(10):
        assert fd[i] == i
        assert i in list(fd.keys())

    # Test insertion
    fd[0] = 10
    assert fd[0] == 10

    # Test insertion of more elements
    for i in range(10, 100):
        fd[i] = i
        assert len(fd) == 10


def testFiniteDictionaryLast():
    fd = FiniteDictionary(10)

    for i in range(10):
        fd[i] = i

        assert fd.last() == i


def testUnravelIndex():
    x = np.empty((1, 11, 100))

    for i in range(np.product(x.shape)):
        expected = np.unravel_index(i, x.shape)
        got = unravel_index(i, x.shape)

        print(expected, got)

        assert_allclose(expected, got)


def testSolve2D():
    A0 = np.random.rand(8, 9, 10, 2, 2)
    A = (A0 + A0.swapaxes(-2, -1)) / 2  # Build symmetric matrix

    B = np.random.rand(8, 9, 10, 2)

    x1 = solve(A, B)
    x2 = np.linalg.solve(A, B)
    assert_allclose(x1, x2)


def testSolve3D():
    A0 = np.random.rand(8, 9, 10, 3, 3)
    A = (A0 + A0.swapaxes(-2, -1)) / 2  # Build symmetric matrix

    B = np.random.rand(8, 9, 10, 3)

    x1 = solve(A, B)
    x2 = np.linalg.solve(A, B)
    assert_allclose(x1, x2)


def test_gradient():
    X = np.random.rand(10, 10, 10)

    for ax in range(3):
        ref = np.gradient(X, axis=ax)
        new = gradient(X, axis=ax)

        assert_allclose(ref, new)


def test_trilinear_interpolation():
    np.random.seed(16091992)
    data = np.random.rand(1, 2, 2, 2)
    grid = [(0, 1)] * 3

    interp = RegularGridInterpolator(grid, data[0])

    ref = []
    new = []
    pos = np.random.rand(10_000, 3)
    ref = interp(pos)
    new = trilinear_interpolation(pos, data)[:, 0]

    assert_allclose(ref, new)


def test_trilinear_interpolation_2():
    data = np.random.rand(1, 2, 2, 2)

    vleft = data[0, 0, 0, 0]
    vright = data[0, 1, 0, 0]

    x = np.linspace(0, 1)
    z = np.zeros_like(x)
    X = np.array([x, z, z]).T

    ref = x * (vright - vleft) + vleft
    exp = trilinear_interpolation(X, data)[:, 0]

    assert_allclose(ref, exp)


def test_measure_hessian():
    np.random.seed(16091992)
    x, y, z = np.meshgrid(*[np.arange(-5, 6)] * 3, indexing="ij")
    data = x**2 + y**2 + z**2 + x * y + y * z + x * z

    def anal_hess(x, y, z):
        tmp = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=np.float64)
        return tmp

    ref = []
    exp = []

    all_pos = []
    for i in range(2, 8):
        for j in range(2, 8):
            for k in range(2, 8):
                all_pos.append([i, j, k])
                X = np.array([i - 5.0, j - 5.0, k - 5.0])
                ref.append(anal_hess(*X))

    X = np.array(all_pos, dtype=np.float64)
    exp = measure_hessian(X, data)

    assert_allclose(ref, exp)


def test_measure_gradient():
    np.random.seed(16091992)
    x, y, z = np.meshgrid(*[np.arange(-5, 6)] * 3, indexing="ij")
    data = x**2 + y**2 + z**2 + x * y + y * z + x * z

    def anal_grad(x, y, z):
        tmp = np.array([2 * x + y + z, x + 2 * y + z, x + y + 2 * z], dtype=np.float64)
        return tmp

    ref = []
    exp = []

    all_pos = []
    for i in range(1, 9):
        for j in range(1, 9):
            for k in range(1, 9):
                all_pos.append([i, j, k])
                X = np.array([i - 5.0, j - 5.0, k - 5.0])
                ref.append(anal_grad(*X))

    X = np.array(all_pos, dtype=np.float64)
    exp = measure_gradient(X, data)

    assert_allclose(ref, exp)
