import numpy as np
from numpy.testing import assert_array_almost_equal
from sanssouci.post_hoc_bounds import max_fp, min_tp, curve_max_fp, min_tdp


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
    assert max_fp(p_values, thr) == 80

    # try with pval = null value
    p_values = np.array([])
    thr = np.array([1.e-4])
    assert max_fp(p_values, thr) == 0  # return 0 here size of p_val ?

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

    # try with pval = null value
    p_values = np.array([])
    thr = np.array([1.e-4])
    assert min_tp(p_values, thr) == 0  # size of pvalues return 0

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thr = np.array([1.e-4])
    assert min_tp(p_values, thr) == 0


def test_min_tdp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thr = np.array([1.e-4])
    assert min_tdp(p_values, thr) == 0.2

    # try with a sequence
    thr = np.array([1.e-5, 1.e-4, 1.e-3])
    assert min_tdp(p_values, thr) == 0.2

    # try with unordered sequence
    thr = np.array([1.e-3, 1.e-4, 1.e-5])
    assert_array_almost_equal(min_tdp(p_values, thr), (0.2, 0.2, 0.2))

    # try with pval = null value
    p_values = np.array([])
    thr = np.array([1.e-4])
    assert min_tdp(p_values, thr) == 0  # size of pvalues return 0

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thr = np.array([1.e-4])
    assert min_tdp(p_values, thr) == 0


def test_curve_max_fp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thr = np.array([1.e-4])
    curve_max_fp_ = curve_max_fp(p_values, thr)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curve_max_fp_ <= max_fp(p_values, thr))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thr)

    # try with a sequence
    thr = np.array([1.e-5, 1.e-4, 1.e-3])
    curve_max_fp_ = curve_max_fp(p_values, thr)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curve_max_fp_ <= max_fp(p_values, thr))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thr)

    # try with unordered sequence
    thr = np.array([1.e-3, 1.e-4, 1.e-5])
    curve_max_fp_ = curve_max_fp(p_values, thr)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curve_max_fp_ <= max_fp(p_values, thr))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thr)

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thr = np.array([1.e-4])
    curve_max_fp_ = curve_max_fp(p_values, thr)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.arange(1, 101))
    assert all(curve_max_fp_ <= max_fp(p_values, thr))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thr)

    # try with size thr > p_values
    rng = np.random.RandomState(42)
    p_values = rng.rand(50)
    p_values[:10] /= 10 ** 6
    thr = rng.rand(100)
    curve_max_fp_ = curve_max_fp(p_values, thr)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(10), np.arange(1, 41)))
    assert all(curve_max_fp_ <= max_fp(p_values, thr))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thr)
