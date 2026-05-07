import numpy as np
from numpy.testing import assert_array_almost_equal
from sanssouci.lambda_calibration import get_permuted_p_values
from sanssouci.lambda_calibration import get_pivotal_stats
from sanssouci.lambda_calibration import get_permuted_p_values_one_sample
from sanssouci.row_welch import row_welch_tests
from sanssouci.reference_families import inverse_linear_template
from scipy import stats
from sklearn.utils.validation import check_random_state
from joblib import Parallel, delayed

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
        p0, inverse_template=inverse_linear_template)
    assert piv_stat.shape == (n_permutations,)
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


# slower version of one sample T-tests
def get_permuted_p_values_one_sample_slow(
        X,
        n_permutations=100,
        seed=None,
        n_jobs=1):
    """
    Get permutation p-values: Get a matrix of p-values under the null
    hypothesis obtained by sign-flipping (one-sample test).

    Parameters
    ----------
    X : array-like of shape (n,p)
        numpy array of size [n,p], containing n observations of p variables
        (hypotheses)
    n_permutations : int
        number of sign-flippings to be performed (default=100)
    n_jobs : int
        number of CPUs used for computation. Default = 1

    Returns
    -------
    pval0 : array-like of shape (n_permutations, p)
        Numpy array whose rows are sorted increasingly.
        The entry i,j corresponds to p_{(j)}(g_i.X) with notation of [1]
        (section 4.5)_
    References
    ----------
    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """

    rng = check_random_state(seed)
    seeds = rng.randint(np.iinfo(np.int32).max, size=n_permutations)
    n, p = X.shape

    # intialise p-values
    pval0 = Parallel(n_jobs=n_jobs)(delayed(
        _compute_permuted_pvalues_1samp)(X, seed=seed_) for seed_ in seeds)

    # Sort each line
    pval0 = np.sort(pval0, axis=1)

    return pval0


def _compute_permuted_pvalues_1samp(X, seed=None):
    """
    Compute randomized p-values for a single sign-flip of the input data
    """
    np.random.seed(seed)
    n, p = X.shape
    X_flipped = (X.T * (2 * np.random.randint(-1, 1, size=n) + 1)).T
    _, permuted_pvals = stats.ttest_1samp(X_flipped, 0)
    return permuted_pvals

rng = np.random.RandomState(42)
n = 100
p = 123
n_permutations = 1000
X = rng.randn(n, p)

pvals = get_permuted_p_values_one_sample(X, n_permutations=n_permutations)
pvals2 = get_permuted_p_values_one_sample_slow(X, n_permutations=n_permutations)

# check equal distribution for all variables 
# (no significant KS test after Bonferroni correction)
# KS test for each variable independently
ks_pvals = np.array([
    stats.ks_2samp(pvals[:, i], pvals2[:, i]).pvalue
    for i in range(pvals.shape[1])
])
assert np.min(ks_pvals) * p > 0.1
