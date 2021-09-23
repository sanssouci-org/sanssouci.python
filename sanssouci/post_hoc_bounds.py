import numpy as np
from scipy.stats import norm


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# POST HOC BOUNDS
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def max_fp(p_values, thresholds, k_min=0):
    """
    Upper bound for the number of false discoveries in a selection

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array of p-values for the selected items
    thresholds : 1D numpy.array
        A 1D numpy array of non-decreasing k-FWER-controlling thresholds

    Returns
    -------

    scalar :
        A post hoc upper bound on the number of false discoveries in the
        selection

    See Also
    --------

    sanssouci.min_tp, sanssouci.curve_max_fp

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """

    # make sure that thresholds is sorted
    if np.linalg.norm(thresholds - np.sort(thresholds)) \
       > 0.0001:
        thresholds = np.sort(thresholds)
        print("The input 'thresholds' was not sorted -> this is done now")

    # do the job
    subset_size = p_values.shape[0]
    template_size = thresholds.shape[0]
    size = np.min([subset_size, template_size])

    if size < 1:
        return 0

    if k_min > 0:
        
    seq_k = np.arange(size)

    # k-FWER control for k>subset_size is useless
    # (will yield bound > subset_size)
    thresholds = thresholds[seq_k]

    p_values = np.sort(p_values)
    cutoffs = np.searchsorted(p_values, thresholds)
    card = size - cutoffs

    return np.min([subset_size, (card + seq_k).min()])


def min_tp(p_values, thresholds):
    """
    Lower bound for the number of true discoveries in a selection

    Parameters
    -----------

    p_values : 1D numpy.array
        A 1D numpy array of p-values for the selected items
    thresholds : 1D numpy.array
        A 1D numpy array of non-decreasing k-FWER-controlling thresholds

    Returns
    -------

    scalar :
        A Lower bound on the number of true discoveries in the selection

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """

    return p_values.shape[0] - max_fp(p_values, thresholds)


def min_tdp(p_values, thresholds):
    """Lower bound for the proportion of true discoveries in a selection
    Lower bound for the proportion of true discoveries in a selection

    Parameters
    -----------

    p_values : 1D numpy.array
        A 1D numpy array of p-values for the selected items
    thresholds : 1D numpy.array
        A 1D numpy array of non-decreasing k-FWER-controlling thresholds

    Returns
    -------

    scalar :
        A Lower bound on the proportion of true discoveries in the selection

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """
    if len(p_values) == 0:
        return 0
    else:
        return min_tp(p_values, thresholds) / len(p_values)


def curve_max_fp(p_values, thresholds):
    """
    Upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all $p$ p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of $K$ JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    numpy.array :
        A vector of size p giving an joint upper confidence bound on the
        number of false discoveries among the $k$ most significant items for
        all k in \{1,\ldots,m\}

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """

    #  make sure that p_values and thresholds are sorted
    if np.linalg.norm(p_values - p_values[np.argsort(p_values)]) > 0.0001:
        p_values = p_values[np.argsort(p_values)]
        print("The input p-values were not sorted -> this is done now")

    if np.linalg.norm(thresholds - thresholds[np.argsort(thresholds)]) \
       > 0.0001:
        thresholds = thresholds[np.argsort(thresholds)]
        print("The input 'thresholds' were not sorted -> this is done now")

    # do the job
    p = p_values.shape[0]
    k_max = thresholds.shape[0]

    if k_max < p:
        thresholds = np.concatenate((thresholds, thresholds[-1] *
                                    np.ones(p - k_max)))
        k_max = thresholds.shape[0]

    K = np.ones(p) * (k_max)
    # K[i] = number of k/ T[i] <= s[k] = BB in 'Mein2006'
    Z = np.ones(k_max) * (p)
    # Z[k] = number of i/ T[i] >  s[k] = cardinal of R_k
    # 'K' and 'Z' are initialized to their largest possible value,
    #       ie 'p' and 'k_max', respectively

    kk = 0
    ii = 0

    while (kk < k_max) and (ii < p):
        if thresholds[kk] >= p_values[ii]:
            K[ii] = kk
            ii += 1
        else:
            Z[kk] = ii
            kk += 1

    max_fp_ = np.zeros(p)
    indices = np.where(K > 0)[0]
    A = Z - np.arange(0, k_max)

    K_ww = K[K > 0].astype(np.int)
    cummax_A = A.copy()
    for i in range(1, cummax_A.shape[0]):
        cummax_A[i] = np.max([cummax_A[i - 1], cummax_A[i]])

    cA = cummax_A[K_ww - 1]  # cA[i] = max_{k<K[i]} A[k]

    max_fp_[K > 0] = np.min(
        np.concatenate(((indices + 1 - cA).reshape(1, -1),
                        (K[K > 0]).reshape(1, -1)),
                       axis=0), axis=0)

    return max_fp_


def curve_min_tdp(p_values, thresholds):
    """
    Lower TDP bounds among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all $p$ p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of $K$ JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    numpy.array :
        A vector of size p giving an joint lower confidence bound on the
        true discovery proportion among the $k$ most significant items for
        all k in \{1,\ldots,m\}

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """
    p = p_values.shape[0]
    range = np.arange(1, p + 1)  # use this to shorten following line
    return (range - curve_max_fp(p_values, thresholds)) / range


def find_largest_region(p_values, thresholds, tdp, masker=None):
    """
    Find largest FDP controlling region.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly
    tdp : float
        True discovery proportion
    masker: NiftiMasker
        masker used on current data

    Returns
    -------

    z_unmasked_cal : nifti image of z_values of the FDP controlling region
    region_size : size of TDP controlling region

    """
    z_map_ = norm.isf(p_values)

    res = curve_min_tdp(p_values, thresholds)
    region_size = len(res[res > tdp])
    pval_cutoff = sorted(p_values)[region_size - 1]
    z_cutoff = norm.isf(pval_cutoff)

    if masker is not None:
        z_to_plot = np.copy(z_map_)
        z_to_plot[z_to_plot < z_cutoff] = 0
        z_unmasked_cal = masker.inverse_transform(z_to_plot)
        return z_unmasked_cal, region_size

    return region_size


def _compute_hommel_value(z_vals, alpha):
    """Compute the All-Resolution Inference hommel-value
    Function taken from nilearn.glm
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)

    if len(p_vals) == 1:
        return p_vals[0] > alpha
    if p_vals[0] > alpha:
        return n_samples
    slopes = (alpha - p_vals[: - 1]) / np.arange(n_samples, 1, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(n_samples + (alpha - slope * n_samples) / slope)
    return np.minimum(hommel_value, n_samples)
