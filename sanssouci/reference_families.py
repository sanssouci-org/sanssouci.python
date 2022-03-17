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
