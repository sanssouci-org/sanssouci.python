import numpy as np
from numpy.testing import assert_array_almost_equal
from sanssouci.lambda_calibration import get_permuted_p_values
from sanssouci.lambda_calibration import calibrate_jer
from sanssouci.lambda_calibration import get_pivotal_stats
from sanssouci.lambda_calibration import get_pivotal_stats_shifted
from sanssouci.lambda_calibration import get_permuted_p_values_one_sample
from sanssouci.row_welch import row_welch_tests
from sanssouci.reference_families import linear_template
from sanssouci.reference_families import inverse_linear_template
from sanssouci.reference_families import inverse_shifted_linear_template
from sanssouci.reference_families import beta_template
from scipy import stats
from pytest import warns


def test_get_permuted_p_values():
    rng = np.random.RandomState(42)
    n = 20
    p = 50
    X = rng.randn(n, p)
    n_permutations = 100
    categ = rng.randint(2, size=n)
    pvals = get_permuted_p_values(
        X,
        categ,
        n_permutations=n_permutations,
        row_test_fun=stats.ttest_ind
    )
    assert pvals.shape == (n_permutations, p)
    assert pvals.min() > 1.e-7
    assert pvals.max() <= 1
    assert np.sum(pvals < .1) < n_permutations * p * .12

    stats_ = [stats.ks_2samp, stats.bartlett, stats.ranksums, stats.kruskal]
    for stat in stats_:
        pvals = get_permuted_p_values(
            X,
            categ,
            n_permutations=n_permutations,
            row_test_fun=stat
        )
        assert pvals.min() > 1.e-7
        assert pvals.max() <= 1
        assert np.sum(pvals < .1) < n_permutations * p * .12

    pvals = get_permuted_p_values(
        X,
        categ,
        n_permutations=n_permutations,
        row_test_fun=row_welch_tests
    )
    assert pvals.shape == (n_permutations, p)
    assert pvals.min() > 1.e-7
    assert pvals.max() <= 1
    assert np.sum(pvals < .1) < n_permutations * p * .12


def test_get_permuted_p_values_one_sample():
    rng = np.random.RandomState(42)
    n = 20
    p = 50
    X = rng.randn(n, p)
    n_permutations = 100
    pvals = get_permuted_p_values_one_sample(X, n_permutations=n_permutations)
    assert pvals.shape == (n_permutations, p)
    assert (pvals == np.sort(pvals, axis=1)).all()
    assert pvals.min() > 1.e-7
    assert pvals.max() <= 1
    assert np.sum(pvals < .1) < n_permutations * p * .12


def test_get_pivotal_stat():
    n_permutations = 500
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(n_permutations, p), 1)

    piv_stat = get_pivotal_stats(
        p0,
        inverse_template=inverse_linear_template)
    piv_stat0 = get_pivotal_stats_shifted(
        p0,
        inverse_template=inverse_shifted_linear_template,
        k_min=0)
    assert_array_almost_equal(piv_stat, piv_stat0)

    assert piv_stat.shape == (B,)
    assert piv_stat.min() > 1.e-7
    assert piv_stat.max() < p0.max() * p
    assert isinstance(piv_stat, np.ndarray)
    tk_inv_all = np.array([inverse_linear_template(p0[:, i], i + 1, p)
                          for i in range(p)]).T
    assert_array_almost_equal(piv_stat, np.min(tk_inv_all[:, :p], axis=1))

    piv_stat = get_pivotal_stats(
        p0, inverse_template=inverse_linear_template, k_max=1)
    assert piv_stat.shape == (n_permutations,)
    assert piv_stat.min() > 1.e-7
    assert piv_stat.max() < p0.max() * p
    assert isinstance(piv_stat, np.ndarray)
    tk_inv_all = np.array([inverse_linear_template(p0[:, i], i + 1, p)
                          for i in range(p)]).T
    assert_array_almost_equal(piv_stat, np.min(tk_inv_all[:, :1], axis=1))

    piv_stat = get_pivotal_stats(
        p0, inverse_template=inverse_linear_template, k_max=p)
    assert piv_stat.shape == (n_permutations,)
    assert piv_stat.min() > 1.e-7
    assert piv_stat.max() < p0.max() * p
    assert isinstance(piv_stat, np.ndarray)
    tk_inv_all = np.array(
        [inverse_linear_template(p0[:, i], i + 1, p) for i in range(p)]).T
    assert_array_almost_equal(piv_stat, np.min(tk_inv_all[:, :p], axis=1))

    piv_statB = get_pivotal_stats(
        p0, inverse_template=inverse_linear_template, k_max=p)
    piv_statAll = get_pivotal_stats(
        p0, inverse_template=inverse_linear_template, k_max=-1)
    assert_array_almost_equal(piv_statB, piv_statAll)


def test_coherence_calibration():
    """
    Testing whether calibration via dichotomy and pivotal
    stats are equivalent.
    """
    alpha = 0.05
    B = 500
    p = 50
    seed = 42
    np.random.seed(seed)
    pval0 = np.random.uniform(size=(B, p))

    # Calibrate by dichotmy (calibrate_jer) using Simes template
    alphas = np.linspace(0, 1, 1000)
    simes_matrix = np.vstack([linear_template(alpha,
                                              p, p) for alpha in alphas])
    calibrated_tpl = calibrate_jer(alpha, simes_matrix, pval0, k_max=p)

    # Calibrate using pivotal statistics
    piv_stat = get_pivotal_stats(
        pval0,
        inverse_template=inverse_linear_template,
        K=p)
    lambda_quant = np.quantile(piv_stat, alpha)
    np.testing.assert_array_almost_equal(calibrated_tpl[-1],
                                         lambda_quant, decimal=2)

    # case where all candidates control the JER
    alphas = np.linspace(0, 0.001, 100)
    simes_matrix = np.vstack(
        [linear_template(alpha, p, p) for alpha in alphas])
    with warns(UserWarning, match=r"All templates control the JER"):
        calibrated_tpl = calibrate_jer(alpha, simes_matrix, pval0, k_max=p)
    np.testing.assert_array_almost_equal(calibrated_tpl[-1], alphas[-1])

    # case where no candidate controls the JER (using Beta template)
    alphas = np.linspace(0.9, 1, 100)
    tpl_matrix = np.vstack(
        [beta_template(alpha, p, p) for alpha in alphas])
    with warns(
            UserWarning,
            match=r"No suitable template found; Simes is used instead"):
        calibrated_tpl = calibrate_jer(alpha, tpl_matrix, pval0, k_max=p)
    piv_stat = get_pivotal_stats(pval0,
                                 inverse_template=inverse_linear_template, K=p)
    lambda_quant = np.quantile(piv_stat, alpha)
    np.testing.assert_array_almost_equal(calibrated_tpl[-1],
                                         lambda_quant, decimal=2)
