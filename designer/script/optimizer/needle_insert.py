import sys
sys.path.append('./designer/script/')

import numpy as np
import copy

from film import TwoMaterialFilm
from spectrum import BaseSpectrum
from optimizer.grad_helper import stack_f, stack_J, stack_init_params
from tmm.get_insert_jacobi import get_insert_jacobi_simple

MAX_LAYER = 500


def insert_1_layer(
    film: TwoMaterialFilm,
    target_spec_ls: list[BaseSpectrum],
    insert_search_pts=None,
    insert_places: np.array = None,
    show=True
):
    """
    find the layer and position to insert the new layer and update the film
    NOTE: only applies to the no absorption scenario

    Parameters:
        film
        target_specs:
        insert_search_pts:
        insert_places

    Return: None
    """

    if insert_places is not None:
        raise NotImplementedError(
            'specifying insertion places not yet implemented :(')
    if insert_search_pts is None:  # use default search length
        insert_search_pts = (MAX_LAYER // film.get_layer_number() - 1) // 2
    # check
    if insert_search_pts <= 0:
        raise OverflowError(
            'WARNING: Cannot insert due to layer number limitation')

    insert_idx_arr = make_test_insert_film(film, insert_search_pts)

    grad = get_insert_grad(film, target_spec_ls)

    greedy_insert_idx = np.argmin(grad[insert_idx_arr])
    greedy_insert_layer_idx = insert_idx_arr[greedy_insert_idx]
    # remove test layers but keep the best layer (needle insertion)
    film.remove_negative_thickness_layer(exclude=[greedy_insert_layer_idx])

    # check
    if grad[greedy_insert_layer_idx] >= 0:
        raise ValueError(f'WARNING: positive grad everywhere, min grad:',
                         f'{grad[greedy_insert_layer_idx]}')

    return True, grad[greedy_insert_layer_idx]


def get_insert_grad(film: TwoMaterialFilm, target_spec_ls):
    # prepare initial params
    assert len(target_spec_ls) == 1, 'needle only supports single target spectrum for now.'
    target_spec, n_arrs_ls = target_spec_ls[0].get_R(), stack_init_params(film, target_spec_ls)
    # allocate space and calculate J and f
    d = film.get_d()
    total_wl_num = sum([s.get_R().shape[0] for s in target_spec_ls])
    J = np.empty((total_wl_num * 2, d.shape[0]))
    f = np.empty(total_wl_num * 2)  # only R spec: no absorption
    # TODO: refractor, refer to grad_helper.py
    stack_f(
        f,
        n_arrs_ls,
        d,
        target_spec_ls,
    )
    stack_J(
        J,
        n_arrs_ls,
        d,
        target_spec_ls,
        get_J=get_insert_jacobi_simple, # this function only returns wl * 1 (no T spec)
    )

    # find insertion place with largest negative gradient
    grad = np.dot(J.T, f)
    return grad


def make_test_insert_film(film, insert_search_pts, split=False):
    '''
    Add to every layer evenly insert_search_pts test insert layers,
    as a trail film so that insertion grad at those places could be 
    calculated

    '''
    if split is False:
        # ensure not exceed cuda restrictions
        # which is MAX_LAYER layers (inserted) or layer * insert_pts * 2 < MAX_LAYER
        assert film.get_layer_number() * (insert_search_pts * 2 + 1) <= MAX_LAYER, \
            'too many search points.'  # should have been caught earlier

    insert_idx_arr = [j * 2 + i * (2 * insert_search_pts + 1) + 1
                      for i in range(film.get_layer_number())
                      for j in range(insert_search_pts)]  # max: L(2N + 1) - 2
    d_before = film.get_d().copy()
    for i in range(film.get_layer_number()):

        for j in range(insert_search_pts):
            insert_position = 1 / insert_search_pts * d_before[i]
            # insert new layer: zero thickness
            film.insert_layer(
                j * 2 + i * (2 * insert_search_pts + 1), insert_position, 0)

    return insert_idx_arr
