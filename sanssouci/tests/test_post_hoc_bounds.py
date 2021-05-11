import numpy as np
from numpy.testing import assert_array_almost_equal
from sanssouci.post_hoc_bounds import max_fp, min_tp, curve_max_fp


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

    # try with thr = null value
    # thr = np.array([])
    # assert max_fp(p_values, thr) == 100 #return 0

    # try with pval = null value
    # p_values = np.array([])
    # thr = np.array([1.e-4])
    # assert max_fp(p_values, thr) == 0 #return 0 here size of p_val ?

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

    # try with thr = null value
    # thr = np.array([])
    # p_values.shape[0] - max_fp(p_values, thr)
    #  <=> 100 - 0 (max_fp pb)
    # assert min_tp(p_values, thr) == 0 # return 100, every features are signal

    # try with pval = null value
    # p_values = np.array([])
    # thr = np.array([1.e-4])
    # assert min_tp(p_values, thr) == 0 #size of pvalues return 0

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thr = np.array([1.e-4])
    assert min_tp(p_values, thr) == 0


def test_curve_max_fp():
    p_values = np.linspace(1.e-6, 1 - 1.e-6, 100)
    p_values[:20] /= 10 ** 6

    # try with a scalar
    thr = np.array([1.e-4])
    curveMaxFP = curve_max_fp(p_values, thr)
    assert isinstance(curveMaxFP, np.ndarray)
    assert all(curveMaxFP == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curveMaxFP <= max_fp(p_values, thr))
    assert len(curveMaxFP) == len(p_values)
    assert all(curveMaxFP >= 0)
    assert curveMaxFP[-1] == max_fp(p_values, thr)

    # try with a sequence
    thr = np.array([1.e-5, 1.e-4, 1.e-3])
    curveMaxFP = curve_max_fp(p_values, thr)
    assert isinstance(curveMaxFP, np.ndarray)
    assert all(curveMaxFP == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curveMaxFP <= max_fp(p_values, thr))
    assert len(curveMaxFP) == len(p_values)
    assert all(curveMaxFP >= 0)
    assert curveMaxFP[-1] == max_fp(p_values, thr)

    # try with unordered sequence
    thr = np.array([1.e-3, 1.e-4, 1.e-5])
    curveMaxFP = curve_max_fp(p_values, thr)
    assert isinstance(curveMaxFP, np.ndarray)
    assert all(curveMaxFP == np.append(np.zeros(20), np.arange(1, 81)))
    assert all(curveMaxFP <= max_fp(p_values, thr))
    assert len(curveMaxFP) == len(p_values)
    assert all(curveMaxFP >= 0)
    assert curveMaxFP[-1] == max_fp(p_values, thr)

    # try with random sequence
    rng = np.random.RandomState(42)
    p_values = rng.rand(100)
    thr = np.array([1.e-4])
    curveMaxFP = curve_max_fp(p_values, thr)
    assert isinstance(curveMaxFP, np.ndarray)
    assert all(curveMaxFP == np.arange(1, 101))
    assert all(curveMaxFP <= max_fp(p_values, thr))
    assert len(curveMaxFP) == len(p_values)
    assert all(curveMaxFP >= 0)
    assert curveMaxFP[-1] == max_fp(p_values, thr)

    # try with size thr > p_values
    rng = np.random.RandomState(42)
    p_values = rng.rand(50)
    p_values[:10] /= 10 ** 6
    thr = rng.rand(100)
    curveMaxFP = curve_max_fp(p_values, thr)
    assert isinstance(curveMaxFP, np.ndarray)
    assert all(curveMaxFP == np.append(np.zeros(10), np.arange(1, 41)))
    assert all(curveMaxFP <= max_fp(p_values, thr))
    assert len(curveMaxFP) == len(p_values)
    assert all(curveMaxFP >= 0)
    assert curveMaxFP[-1] == max_fp(p_values, thr)
