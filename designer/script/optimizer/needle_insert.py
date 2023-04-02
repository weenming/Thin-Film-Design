import sys
sys.path.append('./designer/script/')

import numpy as np
import copy

from film import FilmSimple
from spectrum import BaseSpectrum
from grad_helper import stack_f, stack_J
from gets.get_insert_jacobi import get_insert_jacobi_simple

MAX_LAYER = 500

def insert_1_layer(
    film: FilmSimple, 
    target_spec_ls: list[BaseSpectrum], 
    insert_search_pts=None, 
    insert_places: np.array=None, 
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
        raise NotImplementedError('specifying insertion places not yet implemented :(')
    if insert_search_pts is None:
        insert_search_pts = (MAX_LAYER // film.get_layer_number() - 1) // 2
    # check
    if insert_search_pts <= 0:
        raise OverflowError('WARNING: Cannot insert due to layer number limitation')
    
    insert_idx_arr = make_test_insert_film(film, insert_search_pts)

    grad = get_insert_grad(film, target_spec_ls)
    
    greedy_insert_idx = np.argmin(grad[insert_idx_arr])
    greedy_insert_layer_idx = insert_idx_arr[greedy_insert_idx]
    # remove test layers but keep the best layer (needle insertion)
    film.remove_negative_thickness_layer(exclude=[greedy_insert_layer_idx])


    # check
    if grad[greedy_insert_layer_idx] >= 0:
        raise ValueError(f'WARNING: positive grad everywhere, min grad:', \
               f'{grad[greedy_insert_layer_idx]}')
    elif grad[greedy_insert_layer_idx] > -1e-5:
        raise ValueError(f'WARNING: insert gradient close to zero, min grad:', \
              f'{grad[greedy_insert_layer_idx]}')
    elif greedy_insert_idx % insert_search_pts in [0, 1]:
        raise ValueError('WARNING: inserted layer is on the edge of a layer', \
              'which may indicate the termination of needle optimization')

    return True, grad[greedy_insert_layer_idx]



def get_insert_grad(film: FilmSimple, target_spec_ls):
    # stack parameters & preparations
    target_spec = np.array([])
    n_arrs_ls = []
    for s in target_spec_ls:
        target_spec = np.append(target_spec, s.get_R())
        target_spec = np.append(target_spec, s.get_T())
        # calculate refractive indices in advance and store

        # In LM optimization this saves time but in needle
        # insertion it does not. Only to stay close to the 
        # implementation in LM descent for reusing code

        n_arrs_ls.append([
            film.calculate_n_array(s.WLS), 
            film.calculate_n_sub(s.WLS), 
            film.calculate_n_inc(s.WLS)
        ])
        
    # allocate space and calculate J and f
    d = film.get_d()
    J = np.empty((target_spec.shape[0], d.shape[0]))
    f = np.empty(target_spec.shape[0]) # only R spec: no absorption
    # TODO: refractor
    M = MAX_LAYER
    for i in range(d.shape[0] // M):
        stack_J(
            J, 
            [[
                y[0][:, i * M: (i + 1) * M], 
                y[1][i * M: (i + 1) * M], 
                y[2][i * M: (i + 1) * M]
            ] for y in n_arrs_ls], # i know it is shit :( 
            d[i * M: (i + 1) * M], 
            target_spec_ls, 
            get_J=get_insert_jacobi_simple
        )
        stack_f(
            f, 
            [[
                y[0][:, i * M: (i + 1) * M], 
                y[1][i * M: (i + 1) * M], 
                y[2][i * M: (i + 1) * M]
            ] for y in n_arrs_ls], 
            d[i * M: (i + 1) * M], 
            target_spec_ls, 
            target_spec
        )

    if d.shape[0] % M > 0:
        last_i = d.shape[0] // M
        stack_J(
            J, 
            [[
                y[0][:, last_i * M:], 
                y[1][last_i * M:], 
                y[2][last_i * M:]
            ] for y in n_arrs_ls], # i know it is shit :( 
            d[last_i * M:], 
            target_spec_ls, 
            get_J=get_insert_jacobi_simple
        )
        stack_f(
            f, 
            [[
                y[0][:, last_i * M:], 
                y[1][last_i * M:], 
                y[2][last_i * M:]
            ] for y in n_arrs_ls], 
            d[last_i * M:], 
            target_spec_ls, 
            target_spec
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
            'too many search points.' # should have been caught earlier
        
    insert_idx_arr = [j * 2 + i * (2 * insert_search_pts + 1) + 1
                        for i in range(film.get_layer_number()) 
                        for j in range(insert_search_pts)] # max: L(2N + 1) - 2
    d_before = film.get_d().copy()
    for i in range(film.get_layer_number()):
        
        for j in range(insert_search_pts):
            insert_position = 1 / insert_search_pts * d_before[i]
            # insert new layer: zero thickness
            film.insert_layer(j * 2 + i * (2 * insert_search_pts + 1), insert_position, 0) 

    return insert_idx_arr
