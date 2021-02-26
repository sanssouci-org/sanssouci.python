import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# POST HOC BOUNDS
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def max_fp(p_values, thr):
    """
    Upper bound for the number of false discoveries in a selection

    Parameters
    ----------
    p_values : 1D numpy.array
        A 1D numpy array of p-values for the selected items
    thr : 1D numpy.array
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

    # make sure that thr is sorted
    if np.linalg.norm(thr - thr[np.argsort(thr)]) > 0.0001:
        thr = thr[np.argsort(thr)]
        print("The input thr was not sorted -> this is done now")

    # do the job
    nS = p_values.shape[0]
    K = thr.shape[0]
    size = np.min([nS, K])

    if size < 1:
        return 0

    seqK = np.arange(size)

    # k-FWER control for k>nS is useless (will yield bound > nS)
    thr = thr[seqK]

    card = np.zeros(thr.shape[0])
    for i in range(thr.shape[0]):
        card[i] = np.sum(p_values > thr[i])
    # card<-sapply(thr,FUN=function(thr){sum(p_values > thr)})

    return np.min([nS, (card + seqK).min()])


def min_tp(p_values, thr):
    """
    Lower bound for the number of true discoveries in a selection

    Parameters
    -----------
    p_values : 1D numpy.array
        A 1D numpy array of p-values for the selected items
    thr : 1D numpy.array
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

    return p_values.shape[0] - max_fp(p_values, thr)


def curve_max_fp(p_values, thr):
    """
    Upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------
    p_values : 1D numpy.array
        A 1D numpy array containing all $p$ p-values,sorted non-decreasingly
    thr : 1D numpy.array
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

    #  make sure that p_values and thr are sorted
    if np.linalg.norm(p_values - p_values[np.argsort(p_values)]) > 0.0001:
        p_values = p_values[np.argsort(p_values)]
        print("The input p-values were not sorted -> this is done now")

    if np.linalg.norm(thr - thr[np.argsort(thr)]) > 0.0001:
        thr = thr[np.argsort(thr)]
        print("The input thr were not sorted -> this is done now")

    # do the job
    p = p_values.shape[0]
    kMax = thr.shape[0]

    if kMax < p:
        thr = np.concatenate((thr, thr[-1] * np.ones(p - kMax)))
        kMax = thr.shape[0]

    K = np.ones(p) * (kMax)
    # K[i] = number of k/ T[i] <= s[k] = BB in 'Mein2006'
    Z = np.ones(kMax) * (p)
    # Z[k] = number of i/ T[i] >  s[k] = cardinal of R_k
    # 'K' and 'Z' are initialized to their largest possible value,
    #       ie 'p' and 'kMax', respectively

    kk = 0
    ii = 0

    while (kk < kMax) and (ii < p):
        if thr[kk] >= p_values[ii]:
            K[ii] = kk
            ii += 1
        else:
            Z[kk] = ii
            kk += 1

    Vbar = np.zeros(p)
    ww = np.where(K > 0)[0]
    A = Z - np.arange(0, kMax)

    K_ww = K[ww].astype(np.int)
    cummax_A = A.copy()
    for i in range(1, cummax_A.shape[0]):
        cummax_A[i] = np.max([cummax_A[i - 1], cummax_A[i]])

    cA = cummax_A[K_ww - 1]  # cA[i] = max_{k<K[i]} A[k]

    Vbar[ww] = np.min(np.concatenate(((ww + 1 - cA).reshape(1, -1),
                                      (K[ww]).reshape(1, -1)),
                                     axis=0),
                      axis=0)

    return Vbar
