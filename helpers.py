import numpy as np
from scipy.stats import poisson

from const import STATE


def p_t(H_val, M_val, k, x0):
    """probability of going to hospital
    depends on the number of vacant beds H_val - M_val
    and logistic function parameters, k (controls steepiness) and x0
    """
    x = H_val - M_val
    if x > 0:
        return 1 / (1 + np.exp(-k * (x - x0)))
    else:
        return 0


def pr_IM_long(T, t, data,  k, x0):
    """
    probability of transitting from I at time T-t to M at time T
    """
    p_T = p_t(data[T, STATE.H], data[T, STATE.M], k, x0)
    if p_T == 0:
        return 0
    p_T_minus_t_list = np.array([p_t(data[T-i, STATE.H], data[T-i, STATE.M], k, x0) for i in range(1, t)])
    p = np.prod(1-p_T_minus_t_list) * p_T
    return p


def truncated_poisson(x, mu, min_x, max_x):
    assert x == int(x)
    x = int(x)
    assert x <= max_x
    all_probas = np.array([poisson.pmf(xi, mu) for xi in range(min_x, max_x+1)])
    all_probas /= all_probas.sum()
    assert np.isclose(all_probas.sum(), 1)
    return all_probas[x-min_x]


def pr_EI_long(t, mu_ei, k):
    assert t >= 1
    return truncated_poisson(t, mu_ei, 1, k)


def pr_MO_long(t, mu_mo, k):
    assert t >= 1
    return truncated_poisson(t, mu_mo, 1, k)
