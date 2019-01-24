from py_extrema.utils import FiniteDictionary, unravel_index, solve
import numpy as np

from numpy.testing import assert_allclose


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
