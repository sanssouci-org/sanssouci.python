import numpy as np
from sanssouci.row_welch import get_summary_stats
from sanssouci.row_welch import suff_welch_test
from sanssouci.row_welch import row_welch_tests
from numpy.testing import assert_array_almost_equal


def test_get_summary_stats():
    rng = np.random.RandomState(42)
    n = 20
    p = 50
    X = rng.randn(n, p)
    labels = rng.randint(2, size=n)
    summary = get_summary_stats(X, labels)

    assert isinstance(summary, dict)
    assert len(summary) == 2
    assert summary.keys() == set(labels)
    for key, values in summary.items():
        assert len(values) == 5
        assert isinstance(values, dict)
        assert values.keys() == {'sum', 'sum2', 'n', 'mean', 'sd'}

    assert summary[0]["n"] == n - summary[1]["n"]
    assert_array_almost_equal(summary[0]["sum"] + summary[1]["sum"],
                              np.sum(X, axis=0))
    assert_array_almost_equal(summary[0]["sum2"] + summary[1]["sum2"],
                              np.sum(X * X, axis=0))


def test_suff_welch_test():
    rng = np.random.RandomState(42)
    p = 50
    n = 20
    nx = 10
    ny = n - nx

    mx = rng.random_sample(p) - 1
    sx = rng.randn(p) * 0.5 + 1
    my = rng.random_sample(p)
    sy = rng.randn(p) * 0.1 + 2

    welch = suff_welch_test(mx, my, sx, sy, nx, ny)
    assert isinstance(welch, dict)
    assert len(welch) == 3
    assert welch.keys() == {'statistic', 'parameter', 'p_value'}
    for key, value in welch.items():
        assert len(value) == p
        assert isinstance(value, np.ndarray)


def test_row_welch_tests():
    rng = np.random.RandomState(42)
    n = 20
    p = 50
    X = rng.randn(n, p)
    labels = rng.randint(2, size=n)

    welch = row_welch_tests(X, labels)

    assert isinstance(welch, dict)
    assert len(welch) == 4
    assert welch.keys() == {'statistic', 'parameter', 'p_value', 'meanDiff'}
    for key, value in welch.items():
        assert len(value) == p
        assert isinstance(value, np.ndarray)
