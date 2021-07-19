import numpy as np
from scipy import stats

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ROW WELCH TESTS
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_summary_stats(X, labels):
    """
    Convert a matrix of observations labelled into categories into summary
    statistics for each category

    The following statistics are calculated: sums, sums of squares, means,
    standard deviations, sample sizes

    Parameters
    ----------

    X : array-like of shape (n, p)
        Numpy array matrix whose columns correspond to the p variables
        and rows to the n observations
    labels :  array-like of shape (n, )
        A numpy array of size n representing the label of each
        observation, in {0, 1}

    Returns
    -------

    test : dict of float arrays of shape (p,)
        A dictionary containing the above-described summary statistics for
        each category
    """

    labels_set = set(labels)

    res = {}

    for lab in labels_set:
        X_lab = X[labels == lab]
        sum_lab = np.sum(X_lab, axis=0)
        sum_squares_lab = np.sum(X_lab * X_lab, axis=0)
        n_lab = np.sum(labels == lab)
        mean_lab = sum_lab / n_lab
        std_lab = np.sqrt((sum_squares_lab -
                           ((sum_lab * sum_lab) / n_lab)) / (n_lab - 1))
        res[lab] = {"sum": sum_lab, "sum2": sum_squares_lab, "n": n_lab,
                    "mean": mean_lab, "sd": std_lab}

    return res


def suff_welch_test(mean_x, mean_y, std_x, std_y, n_x, n_y):
    """
    Welch test from sufficient statistics

    Parameters
    ----------

    mean_x : array-like
        A numeric value or vector, the sample average for condition "x"
    mean_y : array-like
        A numeric value or vector of the same length as 'mean_x', the sample
        average for condition "y"
    std_x : array-like
        A numeric value or vector of the same length as 'mean_x', the standard
        deviation for condition "x"
    std_y : array-like
        A numeric value or vector of the same length as 'mean_x', the standard
        deviation for condition "y"
    n_x : array-like
        A numeric value or vector of the same length as 'mean_x', the sample
        size for condition "x"
    n_y : array-like
        A numeric value or vector of the same length as 'mean_x', the sample
        size for condition "y"

    Returns
    -------

    test : dict of float arrays of shape (p,) with
        statistic: the value of the t-statistic
        parameter:  the degrees of freedom for the t-statistic
        p_value: the p-value for the test

    Notes
    -----

    Note that the alternative hypothesis is "two.sided". It could be extended
    to "greater" or "less" as in the original R code.
    """

    # pre-computations: squared standard error of the mean (sem)
    squared_sem_x = (std_x * std_x) / n_x
    squared_sem_y = (std_y * std_y) / n_y
    squared_sem = squared_sem_x + squared_sem_y

    # test statistic
    stat = (mean_x - mean_y) / np.sqrt(squared_sem)

    # approximate degrees of freedom (Welch-Satterthwaite)
    df = squared_sem * squared_sem /\
        ((squared_sem_x * squared_sem_x / (n_x - 1)) +
         ((squared_sem_y * squared_sem_y) / (n_y - 1)))

    # two-sided p-value
    p_value = 2 * (1 - stats.t.cdf(np.abs(stat), df=df))

    return {"statistic": stat, "parameter": df, "p_value": p_value}


def row_welch_tests(X, labels):
    """
    Welch t-tests for each column of a matrix, intended to be speed efficient

    Note that the alternative hypothesis is "two.sided". It could be extended
    to "greater" or "less" as in the original R code.

    Parameters
    ----------

    X : array-like of shape (n, p)
        Numpy array matrix whose columns correspond to the p variables
         and rows to the n observations
    labels :  array-like of shape (n, )
        A numpy array of size n representing the category of each
        observation, in {0, 1}

    Returns
    -------

    test : dict of float arrays of shape (p,) with
        statistic: the value of the t-statistic
        parameter:  the degrees of freedom for the t-statistic
        p_value: the p-value for the test
        meanDiff: the difference between means

    Notes
    -----

    Note that the alternative hypothesis is "two.sided". It could be extended
    to "greater" or "less" as in the original R code.

    References
    ----------

    ..[1] B. L. Welch (1951), On the comparison of several mean values:
        an alternative approach. Biometrika, 38, 330-336

    """

    summary_stats = get_summary_stats(X, labels)

    Y = summary_stats[0]
    X = summary_stats[1]

    welch = suff_welch_test(X["mean"], Y["mean"], X["sd"],
                            Y["sd"], X["n"], Y["n"])
    welch["meanDiff"] = X["mean"] - Y["mean"]

    return welch
