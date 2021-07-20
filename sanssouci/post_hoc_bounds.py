import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# POST HOC BOUNDS
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def max_fp(p_values, thresholds):
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

    ndarray or scalar :
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

    seq_k = np.arange(size)

    # k-FWER control for k>subset_size is useless
    # (will yield bound > subset_size)
    thresholds = thresholds[seq_k]

    card = np.zeros(thresholds.shape[0])
    for i in range(thresholds.shape[0]):
        card[i] = np.sum(p_values > thresholds[i])

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

    ndarray or scalar :
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
    A = Z - np.arange(0, k_max)

    K_ww = K[K > 0].astype(np.int)
    cummax_A = A.copy()
    for i in range(1, cummax_A.shape[0]):
        cummax_A[i] = np.max([cummax_A[i - 1], cummax_A[i]])

    cA = cummax_A[K_ww - 1]  # cA[i] = max_{k<K[i]} A[k]

    max_fp_[K > 0] = np.min(np.concatenate((
                                    (np.array(K > 0) + 1 - cA).reshape(1, -1),
                                    (K[K > 0]).reshape(1, -1)),
                                    axis=0),
                            axis=0)

    return max_fp_
