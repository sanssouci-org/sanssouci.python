import numpy as np
from sanssouci.reference_families import t_inv_linear, t_linear, t_beta
from sanssouci.reference_families import t_inv_beta
from scipy.stats import beta
from numpy.testing import assert_array_almost_equal


def test_t_inv_linear():
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(p,))
    til = t_inv_linear(p0, 2, p)
    assert (til >= p0).all()
    assert til.min() > 0.01
    assert til.max() < 10
    assert til.mean() > .9
    assert til.mean() < 5

    assert isinstance(til, np.ndarray)
    assert til.shape == p0.shape

    til_full_template = t_inv_linear(p0, p, p)
    assert_array_almost_equal(til_full_template, p0)


def test_t_linear():
    alpha = .05
    k = 5
    m = 10
    t = t_linear(alpha, k, m)
    assert_array_almost_equal(t, alpha * np.arange(1, k + 1) / m)
    assert isinstance(t, np.ndarray)
    assert len(t) == k


def test_t_beta():
    alpha = .05
    k = 5
    m = 10
    t = t_beta(alpha, k, m)
    assert_array_almost_equal(t, beta.ppf(alpha, np.arange(1, k + 1),
                              np.array([m + 1] * k) - np.arange(1, k + 1)))
    assert isinstance(t, np.ndarray)
    assert len(t) == k


def test_t_inv_beta():
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(p,))
    til = t_inv_beta(p0, 2, p)
    assert (til >= p0).all()
    assert til.min() > 0.01
    assert til.max() <= 1
    assert til.mean() > .5
    assert til.mean() <= 1

    assert isinstance(til, np.ndarray)
    assert til.shape == p0.shape
