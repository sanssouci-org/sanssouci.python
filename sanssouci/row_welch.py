import numpy as np
from scipy import stats

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROW WELSH TESTS
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_summary_stats(mat, categ):
    """
    Convert a matrix of observations labelled into categories into summary
    statistics for each category

    The following statistics are calculated: sums, sums of squares, means,
    standard deviations, sample sizes

    Parameters
    ----------

    mat : array-like of shape (n, p)
        Numpy array matrix whose columns correspond to the p variables
        and rows to the n observations
    categ :  array-like of shape (n, )
        A numpy array of size n representing the category of each
        observation, in {0, 1}

    Returns
    -------

    Dict :
        A dictionary containing the above-described summary statistics for
        each category
    """

    cats = set(categ)

    res = {}

    for cc in cats:
        ww = np.where(categ[:] == cc)[0]
        matc = mat[ww, :]
        sumc = np.sum(matc, axis=0)
        sum2c = np.sum(matc * matc, axis=0)
        nc = ww.shape[0]
        mc = sumc / nc
        sc = np.sqrt((sum2c - ((sumc * sumc) / nc)) / (nc - 1))

        res[cc] = {"sum": sumc, "sum2": sum2c, "n": nc, "mean": mc, "sd": sc}

    return res


def suff_welch_test(mx, my, sx, sy, nx, ny):
    """
    Welch test from sufficient statistics

    Parameters
    ----------

    mx : array-like
        A numeric value or vector, the sample average for condition "x"
    my : array-like
        A numeric value or vector of the same length as 'mx', the sample
        average for condition "y"
    sx : array-like
        A numeric value or vector of the same length as 'mx', the standard
        deviation for condition "x"
    sy : array-like
        A numeric value or vector of the same length as 'mx', the standard
        deviation for condition "y"
    nx : array-like
        A numeric value or vector of the same length as 'mx', the sample
        size for condition "x"
    ny : array-like
        A numeric value or vector of the same length as 'mx', the sample
        size for condition "y"

    Returns
    -------

    Dict:
        Dictionary with elements
        statistic: the value of the t-statistic
        parameter:  the degrees of freedom for the t-statistic
        p_value: the p-value for the test

    Notes
    -----

    Note that the alternative hypothesis is "two.sided". It could be extended
    to "greater" or "less" as in the original R code.
    """

    # pre-computations
    sse_x = (sx * sx) / nx
    sse_y = (sy * sy) / ny
    sse = sse_x + sse_y
    sse2 = sse * sse

    # test statistic
    stat = (mx - my) / np.sqrt(sse)

    # approximate degrees of freedom (Welch-Satterthwaite)
    df = sse2 / ((sse_x * sse_x / (nx - 1)) + ((sse_y * sse_y) / (ny - 1)))

    # p-value
    pval = 2 * (1 - stats.t.cdf(np.abs(stat), df=df))

    return {"statistic": stat, "parameter": df, "p_value": pval}


def row_welch_tests(mat, categ):
    """
    Welch t-tests for each column of a matrix, intended to be speed efficient

    Note that the alternative hypothesis is "two.sided". It could be extended
    to "greater" or "less" as in the original R code.

    Parameters
    ----------

    mat : array-like of shape (n, p)
        Numpy array matrix whose columns correspond to the p variables
         and rows to the n observations
    categ :  array-like of shape (n, )
        A numpy array of size n representing the category of each
        observation, in {0, 1}

    Returns
    -------

    Dict :
        Dictionary with elements
        statistic: the value of the t-statistic
        parameter:  the degrees of freedom for the t-statistic
        p_value: the p-value for the test
        meanDiff: the log Fold Change between category 0 and 1

    Notes
    -----

    Note that the alternative hypothesis is "two.sided". It could be extended
    to "greater" or "less" as in the original R code.

    References
    ----------

    ..[1] B. L. Welch (1951), On the comparison of several mean values:
        an alternative approach. Biometrika, 38, 330-336

    """

    # n = mat.shape[0]
    # p = mat.shape[1]

    sstats = get_summary_stats(mat, categ)

    Y = sstats[0]
    X = sstats[1]

    swt = suff_welch_test(X["mean"], Y["mean"], X["sd"],
                          Y["sd"], X["n"], Y["n"])
    swt["meanDiff"] = X["mean"] - Y["mean"]

    return swt
