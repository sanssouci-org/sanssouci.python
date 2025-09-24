import numpy as np
from scipy.stats import beta


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# REFERENCE FAMILIES
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def linear_template(alpha, k, m):
    """
    Parameters
    ----------
    alpha : float
        confidence level in [0, 1]
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


def linear_template_kmin(k, m, k_min, alpha):
    """
    Parameters
    ----------
    alpha : float
        confidence level in [0, 1]
    k : int
        number of rejection sets in linear template
    m : int
        number of hypotheses
    Returns
    -------
    template : array of shape (k,)
    """
    template = alpha * np.arange(1, k + 1) / m
    template[: k_min] = np.zeros(k_min)

    return template


def inverse_linear_template_kmin(y, k, m, k_min=0):
    """
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
    if k <= k_min:
        return np.array([np.inf] * y.shape[0])
    else:
        return inverse_linear_template(y, k, m)


def beta_template(alpha, k, m):
    """
    Parameters
    ----------
    alpha : float
        confidence level in [0, 1]
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



def inverse_shifted_linear_template(y, k, m, k_min):
    """
    Parameters
    ----------
    y : array of floats of shape (B, )
        values to apply template to
    k : int
        index of rejection set in beta template
    m : int
        number of hypotheses
    k_min : 
        parameter that defines the shift of the template. The template is zero for k <= k_min.

    Returns
    -------
    estimated_template : array of same shape as y
    """
    if k - k_min <= 0:
        return np.array([np.inf] * y.shape[0])
    else:
        return y * (m - k_min)/(k - k_min)


def shifted_linear_template(k, m, k_min, lbd):
    """
    Parameters
    ----------
    k : int
        number of rejection sets in Beta template
    m : int
        number of hypotheses
    k_min : int
        parameter that defines the shift of the template. The template is zero for k <= k_min.
    lbd : float
        parameter in [0, 1], selected to satisfy statistical properties.
    Returns
    -------
    template : array of shape (k,)
    """
    return lbd * np.array([max(0, (j - k_min)/(m - k_min)) for j in range(1, k + 1)])
