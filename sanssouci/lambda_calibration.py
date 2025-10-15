import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sanssouci.row_welch import row_welch_tests
from sanssouci.reference_families import inverse_linear_template
from sanssouci.reference_families import inverse_shifted_linear_template
from sanssouci.reference_families import linear_template
import warnings

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LAMBDA-CALIBRATION
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_permuted_p_values(X, labels, B=100, row_test_fun=stats.ttest_ind):
    """
    Get permutation p-values: Get a matrix of p-values under the null
    hypothesis obtained by repeated permutation of class labels.
    Parameters
    ----------
    X : array-like of shape (n,p)
        numpy array of size [n,p], containing n observations of p variables
        (hypotheses)
    labels : array-like of shape (n,)
        numpy array of size [n], containing n values in {0, 1}, each of them
        specifying the column indices of the first and the second sample.
    B : int
        number of permutations to be performed (default=100)
    row_test_fun : function
        testing function with the same I/O as 'stats.ttest_ind' (default).
        Specifically, must have two lists as inputs (or 1d np.arrays) for
        thecompared data, and the resulting pvalue must be accessed in
        '[test].pvalue' Eligible functions are for instance "stats.ks_2samp",
        "stats.bartlett", "stats.ranksums", "stats.kruskal"
    Returns
    -------
    pval0 : array-like of shape (B, p)
        A numpy array of size [B,p], whose entry i,j corresponds to
        p_{(j)}(g_i.X) with notation of the AoS 2020 paper cited below
        (section 4.5) [1]_
    References
    ----------
    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """

    # Init
    n, p = X.shape

    # Step 1: calculate $p$-values for B permutations of the class assignments

    # 1.1: Intialise all vectors and matrices
    shuffled_labels = labels.copy()

    all_shuffled_labels = np.zeros((B, n))
    for bb in range(B):
        np.random.shuffle(shuffled_labels)
        all_shuffled_labels[bb] = shuffled_labels

    # 1.2: calculate the p-values
    pval0 = np.zeros([B, p])

    if row_test_fun == row_welch_tests:  # row Welch Tests (parallelized)
        for b in range(B):
            permuted_test_result = row_welch_tests(X, all_shuffled_labels[b])
            pval0[b] = permuted_test_result['p_value']
    else:                     # standard scipy tests
        for b in range(B):
            s0 = np.where(all_shuffled_labels[b] == 0)[0]
            s1 = np.where(all_shuffled_labels[b] == 1)[0]

            for ii in range(p):
                rwt = row_test_fun(X[s0, ii], X[s1, ii])
                # Welch test with scipy -> rwt=stats.ttest_ind(X[s0, ii],
                # X[s1, ii], equal_var=False)

                pval0[b, ii] = rwt.pvalue

    # Step 2: sort each column
    pval0 = np.sort(pval0, axis=1)

    return pval0


def get_permuted_p_values_one_sample(X, B=100, seed=None, n_jobs=1):
    """
    Get permutation p-values: Get a matrix of p-values under the null
    hypothesis obtained by sign-flipping (one-sample test).
    Parameters
    ----------
    X : array-like of shape (n,p)
        numpy array of size [n,p], containing n observations of p variables
        (hypotheses)
    B : int
        number of sign-flippings to be performed (default=100)
    n_jobs : int
        number of CPUs used for computation. Default = 1

    Returns
    -------
    pval0 : array-like of shape (B, p)
        A numpy array of size [B,p], whose rows are sorted increasingly.
        The entry i,j corresponds to p_{(j)}(g_i.X) with notation of [1]
        (section 4.5)_
    References
    ----------
    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """

    rng = check_random_state(seed)
    seeds = rng.randint(np.iinfo(np.int32).max, size=B)
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


def get_pivotal_stats(p0, inverse_template=inverse_linear_template, K=-1):
    """Get pivotal statistic

    Parameters
    ----------

    p0 :  array-like of shape (B, p)
        A numpy array of size [B,p] of null p-values obtained from
        B permutations for p hypotheses.
    inverse_template : function
        A function with the same I/O as inverse_template_linear
    K :  int
        For JER control over 1:K, i.e. joint control of all k-FWER, k<= K.
        Automatically set to p if its input value is < 0.

    Returns
    -------

    array-like of shape (B,)
        A numpy array of of size [B]  containing the pivotal statitics, whose
        j-th entry corresponds to psi(g_j.X) with notation of the AoS 2020
        paper cited below (section 4.5) [1]_

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """
    # Sort permuted p-values
    p0 = np.sort(p0, axis=1)

    # Step 3: apply template function
    # For each feature p, compare sorted permuted p-values to template
    B, p = p0.shape
    tk_inv_all = np.array([inverse_template(p0[:, i], i + 1, p)
                           for i in range(p)]).T

    if K < 0:
        K = tk_inv_all.shape[1]  # tkInv_all.shape[1] is equal to p

    # Step 4: report min for each row
    pivotal_stats = np.min(tk_inv_all[:, :K], axis=1)

    return pivotal_stats


def get_pivotal_stats_shifted(p0,
                              inverse_template=inverse_shifted_linear_template,
                              K=-1,
                              k_min=0):
    """Get pivotal statistic

    Parameters
    ----------

    p0 :  array-like of shape (B, p)
        A numpy array of size [B,p] of null p-values obtained from
        B permutations for p hypotheses.
    inverse_template : function
        A function with the same I/O as inverse_template_linear
    K :  int
        For JER control over 1:K, i.e. joint control of all k-FWER, k<= K.
        Automatically set to p if its input value is < 0.
    k_min : int
        parameter that defines the shift of the template.
        The template is zero for k <= k_min.

    Returns
    -------

    array-like of shape (B,)
        A numpy array of of size [B]  containing the pivotal statitics, whose
        j-th entry corresponds to psi(g_j.X) with notation of the AoS 2020
        paper cited below (section 4.5) [1]_

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """
    # Sort permuted p-values
    p0 = np.sort(p0, axis=1)

    # Step 3: apply template function
    # For each feature p, compare sorted permuted p-values to template
    B, p = p0.shape
    tk_inv_all = np.array([inverse_template(p0[:, i], i + 1, p, k_min=k_min)
                           for i in range(p)]).T

    if K < 0:
        K = tk_inv_all.shape[1]  # tkInv_all.shape[1] is equal to p

    # Step 4: report min for each row
    pivotal_stats = np.min(tk_inv_all[:, k_min: K], axis=1)

    return pivotal_stats


def estimate_jer(template, pval0, k_max, k_min=0):
    """
    Compute empirical JER for a given template and permuted p-values
    """

    B, p = pval0.shape
    id_ranks = np.tile(np.arange(0, p), (B, 1))

    cutoffs = np.searchsorted(template, pval0, side='right')

    signs = np.sign(id_ranks - cutoffs)
    sgn_trunc = signs[:, k_min: k_max]
    JER = np.sum([np.any(sgn_trunc[perm] >= 0) for perm in range(B)]) / B

    return JER


def calibrate_jer(alpha, learned_templates, pval0, k_max, min_dist=1, k_min=0):
    """
    For a given risk level, calibrate the method on learned templates by
    dichotomy. This is equivalent to calibrating using pivotal stats but does
    not require the availability of a closed form inverse template.

    Parameters
    ----------

    alpha : float
        confidence level in [0, 1]
    learned_templates : array of shape (B', p)
        learned templates for B' permutations and p voxels
    pval0 :  array of shape (B, p)
        permuted p-values
    k_max : int
        template size
    min_dist : int
        minimum distance to stop iterating dichotomy. Default = 1.
    k_min : int
        parameter that defines the shift of the template.
        The template is zero for k <= k_min.

    Returns
    -------

    thr : list of length k_max
        Threshold family chosen by calibration

    """

    # Sort permuted p-values
    pval0 = np.sort(pval0, axis=1)

    B, p = learned_templates.shape
    low, high = 0, B - 1

    if estimate_jer(learned_templates[high],
                    pval0, k_max, k_min=k_min) <= alpha:
        # check if all learned templates control the JER
        warnings.warn("All templates control the JER:\
                       choice may be conservative")
        return learned_templates[high][:k_max]

    if estimate_jer(learned_templates[low],
                    pval0, k_max, k_min=k_min) >= alpha:
        warnings.warn("No suitable template found; Simes is used instead")
        # check if any learned templates controls the JER
        # if not, return calibrated Simes
        piv_stat = get_pivotal_stats(pval0, K=k_max)
        lambda_quant = np.quantile(piv_stat, alpha)
        simes_thr = linear_template(lambda_quant, k_max, p)
        return simes_thr

    while high - low > min_dist:
        mid = int((high + low) / 2)
        lw = (
            estimate_jer(learned_templates[low], pval0, k_max, k_min=k_min)
            - alpha
        )
        md = (
            estimate_jer(learned_templates[mid], pval0, k_max, k_min=k_min)
            - alpha
        )

        if md == 0:
            return learned_templates[mid][:k_max]
        if lw * md < 0:
            high = mid
        else:
            low = mid
    return learned_templates[low][:k_max]
