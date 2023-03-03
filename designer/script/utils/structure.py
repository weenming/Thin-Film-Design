import numpy as np
from film import FilmSimple

# TODO: optimize metric: calculate structure in smaller or larger thickness?


def diff_simple_film(film1: FilmSimple, film2: FilmSimple, metric='abs', norm=None, wl=750.):
    '''
    Calculate a metric characterizing the difference between two films:
    $\int \|n_1(x) - n_2 (x)\|_1dx$

    Parameters:
        film1: FilmSimple instance
        film2: FilmSimple instance
        norm: the thickness by which the difference metric is normed.
            note that films with different total ot should be punished, but 2 thick films should not
        wl: wl at which refractive index is evaluated
    '''

    n1A, n1B = film1.get_n_A(wl), film1.get_n_B(wl)
    n2A, n2B = film2.get_n_A(wl), film2.get_n_B(wl)
    n1_sub, n2_sub = film1.get_n_sub(wl), film2.get_n_sub(wl)
    d1, d2 = film1.get_d(), film2.get_d()  # array of thicknesses in nm

    if metric == 'abs':
        calculate_diff = _calculate_structure_difference_simple_film_abs
    elif metric == 'RMS':
        calculate_diff = _calculate_structure_difference_simple_film_RMS
    else:
        raise ValueError("bad metric keyword param")

    if np.sum(d1) > np.sum(d2):
        l1_diff = calculate_diff(
            d1, n1A, n1B,
            d2, n2A, n2B,
            n2_sub
        )
    else:
        l1_diff = calculate_diff(
            d2, n2A, n2B,
            d1, n1A, n1B,
            n1_sub
        )

    # norm(?) by the largest possible film
    if norm is None:
        norm = np.max([np.sum(d1), np.sum(d2)])

    l1_diff /= (np.max([n1A, n2A, n1B, n2B]) -
                np.min([n1A, n2A, n1B, n2B])) * norm

    return l1_diff


def _calculate_structure_difference_simple_film_abs(d1, n1A, n1B, d2, n2A, n2B, n_sub):
    '''
    Calculate the difference between two films.
    The metric used is: \int \|n_1(x) - n_2(x)\|_1 dx
    Note that sum(d1) > sum(d2)

    Now this code is shit. It seems to work but is too complicated for me to read.

    Parameters:
        n_sub: the n of substrate for film 2(smaller total thickness)
    '''
    assert np.sum(d1) >= np.sum(d2), "wrong argument passed!"
    diff = 0
    # depth1: accu. thickness till next layer
    i1, i2, depth1, depth2 = 0, 0, d1[0], d2[0]
    depth = 0.

    while depth < np.sum(d1) - 1e-5:
        n1 = [n1A, n1B][i1 % 2]
        n2 = [n2A, n2B][i2 % 2] if i2 < d2.shape[0] else n_sub

        if depth1 < depth2:
            diff += np.abs(n1 - n2) * (depth1 - depth)
            # update
            depth = depth1
            if i1 < d1.shape[0] - 1:
                depth1 += d1[i1 + 1]
                i1 += 1
        elif depth1 > depth2:
            diff += np.abs(n1 - n2) * (depth2 - depth)
            # update
            depth = depth2
            if i2 < d2.shape[0] - 1:
                depth2 += d2[i2 + 1]
                i2 += 1
            else:  # next iter after film2 is calculated to the last layer
                i2 = d2.shape[0]  # set i2 to access n_sub
                depth2 = np.sum(d1)  # so that will enter teh prev. if block
        elif depth1 == depth2:
            diff += np.abs(n1 - n2) * (depth1 - depth)
            # update
            depth = depth1
            if i1 < d1.shape[0] - 1:
                depth1 += d1[i1 + 1]  # last iter, will not run here
                i1 += 1
            if i2 < d2.shape[0] - 1:
                depth2 += d2[i2 + 1]
                i2 += 1
            else:  # next iter after film2 is calculated to the last layer
                i2 = d2.shape[0]  # set i2 to access n_sub
                depth2 = np.sum(d1)  # so that will enter teh prev. if block

    return diff


def _calculate_structure_difference_simple_film_RMS(d1, n1A, n1B, d2, n2A, n2B, n_sub):
    '''
    Calculate the difference between two films.
    The metric used is: \int \|n_1(x) - n_2(x)\|^2 dx
    Note that sum(d1) > sum(d2)

    Now this code is shit. It seems to work but is too complicated for me to read.

    Parameters:
        n_sub: the n of substrate for film 2(smaller total thickness)
    '''
    assert np.sum(d1) >= np.sum(d2), "wrong argument passed!"
    diff = 0
    # depth1: accu. thickness till next layer
    i1, i2, depth1, depth2 = 0, 0, d1[0], d2[0]
    depth = 0.

    while depth < np.sum(d1) - 1e-5:
        n1 = [n1A, n1B][i1 % 2]
        n2 = [n2A, n2B][i2 % 2] if i2 < d2.shape[0] else n_sub

        if depth1 < depth2:
            diff += np.square((n1 - n2) * (depth1 - depth))
            # update
            depth = depth1
            if i1 < d1.shape[0] - 1:
                depth1 += d1[i1 + 1]
                i1 += 1
        elif depth1 > depth2:
            diff += np.square((n1 - n2) * (depth2 - depth))
            # update
            depth = depth2
            if i2 < d2.shape[0] - 1:
                depth2 += d2[i2 + 1]
                i2 += 1
            else:  # next iter after film2 is calculated to the last layer
                i2 = d2.shape[0]  # set i2 to access n_sub
                depth2 = np.sum(d1)  # so that will enter teh prev. if block
        elif depth1 == depth2:
            diff += np.square((n1 - n2) * (depth1 - depth))
            # update
            depth = depth1
            if i1 < d1.shape[0] - 1:
                depth1 += d1[i1 + 1]  # last iter, will not run here
                i1 += 1
            if i2 < d2.shape[0] - 1:
                depth2 += d2[i2 + 1]
                i2 += 1
            else:  # next iter after film2 is calculated to the last layer
                i2 = d2.shape[0]  # set i2 to access n_sub
                depth2 = np.sum(d1)  # so that will enter teh prev. if block

    return np.sqrt(diff)
