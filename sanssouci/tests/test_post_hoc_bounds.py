import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sanssouci.post_hoc_bounds import max_fp, min_tp


def test_max_fp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thr = np.array([1.e-4])
    assert max_fp(p_values, thr) == 80

    # try with a sequence
    thr = np.array([1.e-5, 1.e-4, 1.e-3])
    assert max_fp(p_values, thr) == 80

    # try with unordered sequence
    thr = np.array([1.e-3, 1.e-4, 1.e-5])
    assert_array_almost_equal(max_fp(p_values, thr), (80, 80, 80))

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thr = np.array([1.e-4])
    assert max_fp(p_values, thr) == 100


def test_min_tp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thr = np.array([1.e-4])
    assert min_tp(p_values, thr) == 20

    # try with a sequence
    thr = np.array([1.e-5, 1.e-4, 1.e-3])
    assert min_tp(p_values, thr) == 20

    # try with unordered sequence
    thr = np.array([1.e-3, 1.e-4, 1.e-5])
    assert_array_almost_equal(min_tp(p_values, thr), (20, 20, 20))

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thr = np.array([1.e-4])
    assert min_tp(p_values, thr) == 0
