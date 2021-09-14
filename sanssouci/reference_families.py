import numpy as np
from scipy.stats import beta


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# REFERENCE FAMILIES
# R source code: https://github.com/pneuvial/sanssouci/
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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


def _regularised_inverse_beta_template(y, k, m, reg=100):
    """
    Parameters
    ----------
    y : array of floats of shape (B, )
        values to apply template to
    k : int
        index of rejection set in beta template
    m : int
        number of hypotheses
    reg : float
        scaling factor for beta law

    Returns
    -------

    estimated_template : array of same shape as y

    """
    return beta.cdf(y, k / reg, (m + 1 - k) / reg)


def _regularised_beta_template(alpha, k, m, reg=100):
    """
    Beta template with scaled parameters

    Parameters
    ----------

    alpha : float
        confidence level in [0, 1]
    k : int
        number of rejection sets in Beta template
    m : int
        number of hypotheses
    reg : float
        scaling factor for beta law

    Returns
    -------
    template : array of shape (k,)
    """
    return beta.ppf(alpha, (np.arange(1, k + 1)) / reg,
                    (np.array([m + 1] * k) - np.arange(1, k + 1)) / reg)


def quadratic_template(alpha, k, m):
    """
    Linear template with an additional quadratic term

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

    return alpha * np.arange(1, k + 1) / m + \
        (1 - alpha) * np.square(np.arange(1, k + 1) / m)


def inverse_quadratic_template(y, k, m):
    """
    Parameters
    ----------
    y : array of floats
        values to apply template to
    k : int
        index of rejection set in linear template
    m : int
        number of hypotheses

    Returns
    -------

    estimated_template : array of same shape as y

    """
    if k == m:
        return np.ones(len(y))
    else:
        return (np.array(y) * m ** 2 - k ** 2) / (k * (m - k))
