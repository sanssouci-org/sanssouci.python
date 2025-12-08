import numpy as np
from sanssouci.reference_families import linear_template, inverse_linear_template
from sanssouci.reference_families import beta_template, inverse_beta_template
from sanssouci.reference_families import shifted_linear_template, inverse_shifted_linear_template
from sanssouci.lambda_calibration import calibrate_jer, get_pivotal_stats
from scipy.stats import beta

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_less


def test_linear_template():
    alpha = .05
    k = 5
    m = 10
    t = linear_template(alpha, k, m)
    assert_array_almost_equal(t, alpha * np.arange(1, k + 1) / m)
    assert isinstance(t, np.ndarray)
    assert len(t) == k


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


def test_shifted_linear_template():
    alpha = .05
    k = 50
    m = 100
    k_min = 0
    t = shifted_linear_template(alpha, k, m, k_min)
    assert_array_almost_equal(t, linear_template(alpha, k, m))
    assert isinstance(t, np.ndarray)
    assert len(t) == k

    t1 = shifted_linear_template(alpha, k, m, 10)
    t2 = shifted_linear_template(alpha, k, m, 20)
    assert_array_almost_equal(t1[:10], t1[:10])
    assert_array_less(t2[11:20], t1[11:20])
    assert_array_less(t2[21:], t1[21:])


def test_inverse_shifted_linear_template():
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(p,))
    til = inverse_shifted_linear_template(p0, 10, p, 0)
    assert (til >= p0).all()
    assert_array_almost_equal(til, inverse_linear_template(p0, 10, p))
    assert isinstance(til, np.ndarray)
    assert til.shape == p0.shape

    til_full_template = inverse_shifted_linear_template(p0, p, p, 0)
    assert_array_almost_equal(til_full_template, p0)

    q1 = inverse_shifted_linear_template(p0, 10, p, 1)
    q2 = inverse_shifted_linear_template(p0, 10, p, 2)
    assert_array_less(q1, q2)


def test_coherence_linear_template():
    """
    Testing whether linear_template and inverse_linear_template are actually
    inverse functions of one another. This also tests that calibration via
    dichotomy and pivotal stats are indeed equivalent.
    """
    alpha = 0.05
    B = 100
    p = 50
    seed = 42
    np.random.seed(seed)

    # Calibrate by dichotmy (calibrate_jer) using Simes template
    alphas = np.linspace(0, 1, 1000)
    simes_matrix = np.vstack([linear_template(alpha,
                                              p, p) for alpha in alphas])
    pval0 = np.random.uniform(size=(B, p))
    pval0 = np.sort(pval0, axis=1)
    calibrated_tpl = calibrate_jer(alpha, simes_matrix, pval0, k_max=p)

    # Calibrate using pivotal statistics

    piv_stat = get_pivotal_stats(pval0, K=p)
    lambda_quant = np.quantile(piv_stat, alpha)
    np.testing.assert_array_almost_equal(calibrated_tpl[-1],
                                         lambda_quant, decimal=3)
