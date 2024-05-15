import numpy as np

from film import TwoMaterialFilm, BaseFilm
import matplotlib.pyplot as plt
from design import BaseDesign
import copy


# TODO: optimize metric: calculate structure in smaller or larger thickness?


def diff_simple_film(film1: TwoMaterialFilm, film2: TwoMaterialFilm, metric='abs', norm=None, wl=750.):
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

    n1_ls = film1.calculate_n_array(np.array([wl]))[0, :]
    n2_ls = film2.calculate_n_array(np.array([wl]))[0, :]
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
            d1, n1_ls,
            d2, n2_ls,
            n2_sub
        )
    else:
        l1_diff = calculate_diff(
            d2, n2_ls,
            d1, n1_ls,
            n1_sub
        )

    # norm(?) by the largest possible film
    if norm is None:
        norm = (np.max(np.append(n1_ls, n2_ls)) - np.min(np.append(n1_ls, n2_ls))) * \
            np.max([np.sum(d1), np.sum(d2)])

    l1_diff /= norm

    return l1_diff


def _calculate_structure_difference_simple_film_abs(d1, n1_ls, d2, n2_ls, n_sub):
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
        n1 = n1_ls[i1]
        n2 = n2_ls[i2] if i2 < d2.shape[0] else n_sub

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


def _calculate_structure_difference_simple_film_RMS(d1, n1_ls, d2, n2_ls, n_sub):
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
        n1 = n1_ls[i1]
        n2 = n2_ls[i2] if i2 < d2.shape[0] else n_sub

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


def plot_layer_thickness(film: BaseFilm, n_at_wl=1000,
                         truncate_thickness=float('inf'), ax=None):
    '''
    Plots the refractive index distribution

    Args:
        film (BaseFilm): film structure to show
        n_at_wl (float): middle wl of the first spec
    '''
    # get d vector
    if truncate_thickness != float('inf'):
        d = copy.deepcopy(film).get_d()
        sum = 0
        for j, di in enumerate(d):
            if sum > truncate_thickness:
                break
            sum += di
        d = d[:j]
    else:
        d = film.get_d()

    try:
        spec = film.get_all_spec_list()[0]
        n_at_wl = spec.WLS[spec.WLS.shape[0] // 2, :]
    except IndexError as e:
        print(f'film has no spec. use {n_at_wl} nm')
    n_arr = film.calculate_n_array(np.array([n_at_wl]))[0, :]
    n_inc = film.calculate_n_inc(np.array([n_at_wl]))[0]
    n_sub = film.calculate_n_sub(np.array([n_at_wl]))[0]

    if ax is None:
        new_fig = True
        fig, ax = plt.subplots(1, 1)
    else:
        new_fig = False

    cur_d = 0
    for i in range(d.shape[0]):
        this_n = n_arr[i]
        if i == 0:
            last_n = n_inc
        else:
            last_n = n_arr[i - 1]
        ax.plot([cur_d, cur_d], [this_n, last_n], c='steelblue')
        ax.plot([cur_d, cur_d + d[i]], [this_n, this_n], color='steelblue')
        cur_d += d[i]
    ax.plot([cur_d, cur_d], [this_n, n_sub])
    # ax.set_xlim(0, 20000)

    if new_fig:
        ax.set_xlabel('position / nm')
        ax.set_xlim(0, None)
        ax.set_title(
            f'refractive index distribution at {n_at_wl: .0f} nm')
        
        fig.set_size_inches(6, 1)
        return ax, fig
    


def show_design_process(design: BaseDesign):
    spec = design.film.get_spec()
    n_A = spec.n[spec.WLS.shape[0] // 2, 0]
    n_B = spec.n[spec.WLS.shape[0] // 2, 1]
    n_sub = spec.n_sub[spec.WLS.shape[0] // 2]
    n_arr = [n_A, n_B]

    resolution = 500
    arr = np.zeros((resolution, len(design.training_films)),
                   dtype='complex128') + n_sub

    try:
        l_per_pix = design.get_target_gt() * 2 / arr.shape[0]
    except Exception as e:  # not design for target.
        l_per_pix = design.get_current_gt() / arr.shape[0]

    iter = 0
    for film in design.training_films:
        d = film.get_d()

        for pix in range(arr.shape[0]):
            for i in range(d.shape[0]):
                if d[:i + 1].sum() > pix * l_per_pix:
                    arr[pix, iter] = n_arr[i % 2]
                    break
        iter += 1

    fig, ax = plt.subplots(1, 1)
    s = ax.imshow(arr.real, aspect='auto', cmap='coolwarm',
                  interpolation='none', vmin=1.4, vmax=2.6)
    fig.colorbar(s)
    return arr
