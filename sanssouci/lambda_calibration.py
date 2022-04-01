import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from .row_welch import row_welch_tests
from .reference_families import inverse_linear_template
from .reference_families import linear_template
import warnings

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LAMBDA-CALIBRATION
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_permuted_p_values_one_sample(X, B=100, seed=None, n_jobs=1):
    np.random.seed(seed)
    seeds = np.random.randint(np.iinfo(np.int32).max, size=B)
    n, p = X.shape

    # intialise p-values
    pval0 = Parallel(n_jobs=n_jobs)(delayed(
        _compute_permuted_pvalues_1samp)(X, seed=seed_) for seed_ in seeds)

    # Sort each line
    pval0 = np.sort(pval0, axis=1)

    return pval0


def _compute_permuted_pvalues_1samp(X, seed=None):
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
        j-th entry corresponds to \psi(g_j.X) with notation of the AoS 2020
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


def estimate_jer(template, pval0, k_max):

    """
    Compute empirical JER for a given template and permuted p-values
    """

    B, p = pval0.shape
    id_ranks = np.tile(np.arange(0, p), (B, 1))

    cutoffs = np.searchsorted(template, pval0)

    signs = np.sign(id_ranks - cutoffs)
    sgn_trunc = signs[:, :k_max]
    JER = np.sum([np.any(sgn_trunc[perm] >= 0) for perm in range(B)]) / B

    return JER


def calibrate_jer(alpha, learned_templates, pval0, k_max, min_dist=1):

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

    Returns
    -------

    int : index of template chosen by calibration

    """

    B, p = learned_templates.shape
    low, high = 0, B - 1

    if estimate_jer(learned_templates[high], pval0, k_max) <= alpha:
        # check if all learned templates control the JER
        return learned_templates[high][:k_max]

    if estimate_jer(learned_templates[low], pval0, k_max) >= alpha:
        warnings.warn("No suitable template found; Simes is used instead")
        # check if any learned templates controls the JER
        # if not, return calibrated Simes
        piv_stat = get_pivotal_stats(pval0, K=k_max)
        lambda_quant = np.quantile(piv_stat, alpha)
        simes_thr = linear_template(lambda_quant, k_max, p)
        return simes_thr

    while high - low > min_dist:
        mid = int((high + low) / 2)
        lw = estimate_jer(learned_templates[low], pval0, k_max) - alpha
        md = estimate_jer(learned_templates[mid], pval0, k_max) - alpha
        if md == 0:
            return learned_templates[mid][:k_max]
        if lw * md < 0:
            high = mid
        else:
            low = mid
    return learned_templates[low][:k_max]
