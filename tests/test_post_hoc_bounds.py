import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sanssouci.post_hoc_bounds import max_fp, min_tp, curve_max_fp, min_tdp
from sanssouci.post_hoc_bounds import curve_min_tdp, find_largest_region
from sanssouci.post_hoc_bounds import _compute_hommel_value


def test_max_fp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thresholds = np.array([1.e-4])
    assert max_fp(p_values, thresholds) == 80

    # try with a sequence
    thresholds = np.array([1.e-5, 1.e-4, 1.e-3])
    assert max_fp(p_values, thresholds) == 80

    # try with unordered sequence
    thresholds = np.array([1.e-3, 1.e-4, 1.e-5])
    assert max_fp(p_values, thresholds) == 80

    # try with pval = null value
    p_values = np.array([])
    thresholds = np.array([1.e-4])
    assert max_fp(p_values, thresholds) == 0  # return 0 here size of p_val ?

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thresholds = np.array([1.e-4])
    assert max_fp(p_values, thresholds) == 100

    # corner case: empty vector
    p_values = ()
    thresholds = np.array([1.e-4])
    assert max_fp(p_values, thresholds) == 0


def test_min_tp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thresholds = np.array([1.e-4])
    assert min_tp(p_values, thresholds) == 20

    # try with a sequence
    thresholds = np.array([1.e-5, 1.e-4, 1.e-3])
    assert min_tp(p_values, thresholds) == 20

    # try with unordered sequence
    thresholds = np.array([1.e-3, 1.e-4, 1.e-5])
    assert_array_almost_equal(min_tp(p_values, thresholds), (20, 20, 20))

    # try with pval = null value
    p_values = np.array([])
    thresholds = np.array([1.e-4])
    assert min_tp(p_values, thresholds) == 0  # size of pvalues return 0

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thresholds = np.array([1.e-4])
    assert min_tp(p_values, thresholds) == 0


def test_min_tdp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thresholds = np.array([1.e-4])
    assert min_tdp(p_values, thresholds) == 0.2

    # try with a sequence
    thresholds = np.array([1.e-5, 1.e-4, 1.e-3])
    assert min_tdp(p_values, thresholds) == 0.2

    # try with unordered sequence
    thresholds = np.array([1.e-3, 1.e-4, 1.e-5])
    assert_array_almost_equal(min_tdp(p_values, thresholds), (0.2, 0.2, 0.2))

    # try with pval = null value
    p_values = np.array([])
    thresholds = np.array([1.e-4])
    assert min_tdp(p_values, thresholds) == 0  # size of pvalues return 0

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thresholds = np.array([1.e-4])
    assert min_tdp(p_values, thresholds) == 0


def test_curve_max_fp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thresholds = np.array([1.e-4])
    curve_max_fp_ = curve_max_fp(p_values, thresholds)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curve_max_fp_ <= max_fp(p_values, thresholds))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thresholds)

    # try with a sequence
    thresholds = np.array([1.e-5, 1.e-4, 1.e-3])
    curve_max_fp_ = curve_max_fp(p_values, thresholds)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curve_max_fp_ <= max_fp(p_values, thresholds))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thresholds)

    # try with unordered sequence
    thresholds = np.array([1.e-3, 1.e-4, 1.e-5])
    curve_max_fp_ = curve_max_fp(p_values, thresholds)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curve_max_fp_ <= max_fp(p_values, thresholds))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thresholds)

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thresholds = np.array([1.e-4])
    curve_max_fp_ = curve_max_fp(p_values, thresholds)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.arange(1, 101))
    assert all(curve_max_fp_ <= max_fp(p_values, thresholds))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thresholds)

    # try with size thresholds > p_values
    rng = np.random.RandomState(42)
    p_values = rng.rand(50)
    p_values[:10] /= 10 ** 6
    thresholds = rng.rand(100)
    curve_max_fp_ = curve_max_fp(p_values, thresholds)
    assert isinstance(curve_max_fp_, np.ndarray)
    assert all(curve_max_fp_ == np.append(np.zeros(10), np.arange(1, 41)))
    assert all(curve_max_fp_ <= max_fp(p_values, thresholds))
    assert len(curve_max_fp_) == len(p_values)
    assert all(curve_max_fp_ >= 0)
    assert curve_max_fp_[-1] == max_fp(p_values, thresholds)


def test_curve_min_tdp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6
    # try with a scalar
    thresholds = np.array([1.e-4])
    curve_min_tdp_ = curve_min_tdp(p_values, thresholds)
    tns = 20 / np.arange(21, 101)
    assert isinstance(curve_min_tdp_, np.ndarray)
    assert all(curve_min_tdp_ == np.append(np.ones(20), tns))
    assert all(curve_min_tdp_ >= min_tdp(p_values, thresholds))
    assert len(curve_min_tdp_) == len(p_values)
    assert all(curve_min_tdp_ >= 0)
    assert curve_min_tdp_[-1] == min_tdp(p_values, thresholds)

    # try with a sequence
    thresholds = np.array([1.e-5, 1.e-4, 1.e-3])
    curve_min_tdp_ = curve_min_tdp(p_values, thresholds)
    tns = 20 / np.arange(21, 101)
    assert isinstance(curve_min_tdp_, np.ndarray)
    assert all(curve_min_tdp_ == np.append(np.ones(20), tns))
    assert all(curve_min_tdp_ >= min_tdp(p_values, thresholds))
    assert len(curve_min_tdp_) == len(p_values)
    assert all(curve_min_tdp_ >= 0)
    assert curve_min_tdp_[-1] == min_tdp(p_values, thresholds)


def test_find_largest_region():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6
    thresholds = np.array([1.e-5, 1.e-4, 1.e-3])
    TDP = 0.9
    region_size = find_largest_region(p_values, thresholds, TDP)
    # in the case where len(admissible) == 0
    TDP2 = curve_min_tdp(p_values, thresholds) + 1.e-3
    region_size2 = find_largest_region(p_values, thresholds, TDP2)
    assert region_size >= 0
    assert region_size == 21
    assert region_size2 == 0


def test_compute_hommel_value():
    z_vals1 = np.random.normal(loc=0, scale=1, size=100)
    alpha1 = 2
    with pytest.raises(ValueError, match="alpha should be between 0 and 1"):
        _compute_hommel_value(z_vals1, alpha1)

    z_vals = np.atleast_1d(3)
    hommel_value = _compute_hommel_value(z_vals, 0.05)
    assert hommel_value == 0

    z_vals = np.atleast_1d(1)
    hommel_value = _compute_hommel_value(z_vals, 0.05)
    assert hommel_value == 1

    z_vals = np.random.normal(loc=0, scale=1, size=100)
    hommel_value = _compute_hommel_value(z_vals, 0.1)
    assert hommel_value == len(z_vals)

    hommel_value = _compute_hommel_value(z_vals, 1e-5)
    assert hommel_value == len(z_vals)
