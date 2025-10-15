import numpy as np
from scipy.stats import beta


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# REFERENCE FAMILIES
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def linear_template(alpha, k, m):
    """
    Computes the linear template (Simes template)
    Parameters
    ----------
    alpha : float
        hyperparameter in [0, 1],
        to be chosen to satisfy Joint Error Rate Control.
    k : int
        number of rejection sets in linear template
    m : int
        number of hypotheses
    Returns
    -------
    template : array of shape (k,)
    """
    return alpha * np.arange(1, k + 1) / m


def inverse_linear_template(y, k, m):
    """
    Computes the inverse of the linear template (Simes template)
    Parameters
    ----------
    y : array of floats of shape (B, )
        values to apply template to
    k : int
        index of rejection set in linear template
    m : int
        number of hypotheses
    Returns
    -------
    estimated_template : array of same shape as y
    """

    return y * m / k


def beta_template(alpha, k, m):
    """
    Computes the Beta template
    Parameters
    ----------
    alpha : float
        hyperparameter in [0, 1],
        to be chosen to satisfy Joint Error Rate Control.
    k : int
        number of rejection sets in Beta template
    m : int
        number of hypotheses
    Returns
    -------
    template : array of shape (k,)
    """
    return beta.ppf(alpha, np.arange(1, k + 1),
                    np.array([m + 1] * k) - np.arange(1, k + 1))


def inverse_beta_template(y, k, m):
    """
    Computes the inverse of the Beta template
    Parameters
    ----------
    y : array of floats of shape (B, )
        values to apply template to
    k : int
        index of rejection set in beta template
    m : int
        number of hypotheses
    Returns
    -------
    estimated_template : array of same shape as y
    """
    return beta.cdf(y, k, m + 1 - k)


def shifted_linear_template(alpha, k, m, k_min):
    """
    Computes the shifted linear template (pARI template)
    Parameters
    ----------
    alpha : float
        hyperparameter in [0, 1],
        to be chosen to satisfy Joint Error Rate Control.
    k : int
        number of rejection sets in linear template
    m : int
        number of hypotheses
    k_min : int
        parameter that defines the shift of the template.
        The template is zero for k <= k_min.
    Returns
    -------
    template : array of shape (k,)
    """
    return alpha * np.array([
        max(0, (j - k_min) / (m - k_min))
        for j in range(1, k + 1)
    ])


def inverse_shifted_linear_template(y, k, m, k_min):
    """
    Computes the inverse of the shifted linear template (pARI template)
    Parameters
    ----------
    y : array of floats of shape (B, )
        values to apply template to
    k : int
        index of rejection set in linear template
    m : int
        number of hypotheses
    k_min :
        parameter that defines the shift of the template.
        The template is zero for k <= k_min.

    Returns
    -------
    estimated_template : array of same shape as y
    """
    if k - k_min <= 0:
        return np.array([np.inf] * y.shape[0])
    else:
        return y * (m - k_min) / (k - k_min)
