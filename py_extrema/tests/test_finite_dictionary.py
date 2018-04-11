from py_extrema.extrema import FiniteDictionary


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
