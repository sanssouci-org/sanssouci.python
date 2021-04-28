import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sanssouci.reference_families import t_inv_linear, t_linear
from scipy import stats


def test_t_inv_linear():
    B = 50
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(B, p), 1)
    til = t_inv_linear(p0)
    assert (til >= p0).all()
    assert til.min() > 0.01
    assert til.max() < 10
    assert til.mean() > .9
    assert til.mean() < 1.1
    
    
def test_t_linear():
    alpha = .05
    k = 5
    m = 10
    t = t_linear(alpha, k, m)
    assert t == alpha * k / (m * 1.0)