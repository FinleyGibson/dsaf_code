import matplotlib
import numpy as np
import os
import sys
sys.path.append("/home/finley/phd/code/dsaf/testsuite/")

def format_figures():
    matplotlib.rcParams['font.size'] = 15 ;
    matplotlib.rcParams['legend.fontsize'] = 15
    matplotlib.rcParams['figure.figsize'] = (10, 8)
    

def save_fig(fig, name=None):
    savedirs = ["/home/finley/phd/papers/gecco_2022/gecco_2022_presentation/figures"]
    for d in savedirs:
        fig.savefig(os.path.join(d, name+".png"), dpi=300, facecolor=None, edgecolor=None,
        orientation='portrait', pad_inches=0.12)
        fig.savefig(os.path.join(d, name+".pdf"), bbox_inches = 'tight', pad_inches = 0, dpi=300, facecolor=None, edgecolor=None,
        orientation='portrait')
        
        
def Pareto_split(x_in, maximize: bool = False, return_indices=False):
    """
    separates the data points in data into non-dominated and dominated.

    :param np.ndarray x: the input data (n_points, data_dimension)
    :param bool maximize: True for finding non-dominated points in a
    maximisation problem, else for minimisaiton.
    :param bool return_indices: if True returns the indices of the
    non-dominated and dominate points if False returns the point values
    themselves.
    :return tuple: (non-dominated points, dominated points)
    """
    x = x_in.copy()
    if not return_indices:
        x_orig = x.copy()
    assert x.ndim==2

    if maximize:
        x *= -1

    n_points = x.shape[0]
    is_efficient = np.arange(n_points)
    point_index = 0  # Next index in the is_efficient array to search for
    while point_index<len(x):
        pareto_mask = np.any(x<x[point_index], axis=1)
        pareto_mask[point_index] = True
        is_efficient = is_efficient[pareto_mask]  # Remove dominated points
        x = x[pareto_mask]
        point_index = np.sum(pareto_mask[:point_index])+1

    nondominated_mask = np.zeros(n_points, dtype = bool)
    nondominated_mask[is_efficient] = True
    if return_indices:
        return nondominated_mask, np.invert(nondominated_mask)
    else:
        return x_orig[nondominated_mask], x_orig[np.invert(nondominated_mask)]


def dominates(a: np.ndarray, b: np.ndarray,
              maximize: bool = False,
              strict: bool = True):
    """
    returns True if any of a dominate bi for each element of b,else returns
    False

    usage:
        to check for dominance of b by ANY of a do:
            ans = dominates(a, b)
            - a.shape = (n, m)
            - b.shape = (1, m)
            - ans: bool

        to check for which of b by are dominated by ANY of a do:
            ans = dominates(a, b)
            - a.shape = (n, m)
            - b.shape = (n, m)
            - ans: [bool, bool, ..., bool]

        to check which of a dominate b do:
            ans = [dominates(ai, b) for ai in a]
            - a.shape = (n, m)
            - b.shape = (1, m)
            - ans: [bool, bool, ..., bool]

        to check which of b are dominated by any of a do:
            ans = [dominates(a, bi) for bi in b]
            - a.shape = (n, m)
            - b.shape = (n, m)
            - ans: [bool, bool, ..., bool]

    :param a: np.ndarray (n_points, point_dims)
        dominating query point(s)
    :param b: np.ndarray
        dominated query points (n_points, point_dims)
    :param maximize: bool
        True for finding domination relation in a
        maximisation problem, False for minimisaiton problem.
    :param strict: bool
        if True then computes strict dominance, otherwise allows equal
        value in a given dimension to count as non-dominated
            - swaps < for <=
            - swaps > for >=

    :return bool: True if a dominates b, else returns False"
    """
    if type(a) != np.ndarray:
        a = np.asarray(a)
    if type(b) != np.ndarray:
        b = np.asarray(b)

    if a.ndim < 2:
        a = a.reshape(1, -1)
    if b.ndim < 2:
        b = b.reshape(1, -1)

    if (a.shape[0] != 1) and (b.shape[0] > 1):
        return [dominates(a, bi, maximize, strict) for bi in b]
    elif (a.shape[0] != 1) and (b.shape[0] == 1):
        return bool(np.any([dominates(ai, b, maximize, strict) for ai in a]))

    if not maximize and strict:
        return bool(np.all(a < b, axis=1))
    elif not maximize and not strict:
        return bool(np.all(a <= b, axis=1))
    elif maximize and strict:
        return bool(np.all(a > b, axis=1))
    elif maximize and not strict:
        return bool(np.all(a >= b, axis=1))
    else:
        raise

