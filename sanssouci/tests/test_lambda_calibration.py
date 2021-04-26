import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sanssouci.lambda_calibration import get_perm_p
from scipy import stats


def test_get_perm_p():
    n = 20
    p = 50
    X = np.random.randn(n, p)
    B = 100
    categ = np.random.randint(2, size=n)
    pvals = get_perm_p(X, categ, B=B, row_test_fun=stats.ttest_ind)
    assert pvals.shape == (B, p)
    assert pvals.min() > 1.e-6
    assert pvals.max() <= 1
    assert np.sum(pvals < .1) < B * p * .12

    stats_ = [stats.ks_2samp, stats.bartlett, stats.ranksums, stats.kruskal]
    for stat in stats_:
        pvals = get_perm_p(X, categ, B=B, row_test_fun=stat)
        assert pvals.min() > 1.e-6
        assert pvals.max() <= 1
        assert np.sum(pvals < .1) < B * p * .12

