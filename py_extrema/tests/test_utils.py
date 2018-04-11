from py_extrema.utils import FiniteDictionary, unravel_index
import numpy as np


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

        assert np.all(expected == got)
