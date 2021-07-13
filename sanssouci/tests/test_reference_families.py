import numpy as np
from sanssouci.reference_families import inverse_linear_template
from sanssouci.reference_families import linear_template, beta_template
from sanssouci.reference_families import inverse_beta_template
from scipy.stats import beta
from numpy.testing import assert_array_almost_equal


def test_inverse_linear_template():
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(p,))
    til = inverse_linear_template(p0, 2, p)
    assert (til >= p0).all()
    assert til.min() > 0.01
    assert til.max() < 10
    assert til.mean() > .9
    assert til.mean() < 5

    assert isinstance(til, np.ndarray)
    assert til.shape == p0.shape

    til_full_template = inverse_linear_template(p0, p, p)
    assert_array_almost_equal(til_full_template, p0)


def test_linear_template():
    alpha = .05
    k = 5
    m = 10
    t = linear_template(alpha, k, m)
    assert_array_almost_equal(t, alpha * np.arange(1, k + 1) / m)
    assert isinstance(t, np.ndarray)
    assert len(t) == k


def test_beta_template():
    alpha = .05
    k = 5
    m = 10
    t = beta_template(alpha, k, m)
    assert_array_almost_equal(t, beta.ppf(alpha, np.arange(1, k + 1),
                              np.array([m + 1] * k) - np.arange(1, k + 1)))
    assert isinstance(t, np.ndarray)
    assert len(t) == k


def test_inverse_beta_template():
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(p,))
    til = inverse_beta_template(p0, 2, p)
    assert (til >= p0).all()
    assert til.min() > 0.01
    assert til.max() <= 1
    assert til.mean() > .5
    assert til.mean() <= 1

    assert isinstance(til, np.ndarray)
    assert til.shape == p0.shape
