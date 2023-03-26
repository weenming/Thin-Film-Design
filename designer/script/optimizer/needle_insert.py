import sys
sys.path.append('./designer/script/')

import numpy as np
import copy

from film import FilmSimple
from spectrum import BaseSpectrum
from optimizer.LM_gradient_descent import stack_f, stack_J
from gets.get_insert_jacobi import get_insert_jacobi_simple

MAX_LAYER = 500

def insert_1_layer(
    film: FilmSimple, 
    target_spec_ls: list[BaseSpectrum], 
    insert_search_pts=None, 
    insert_places: np.array=None
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
        assert False, 'not yet implemented :('
    
    insert_search_pts = int(MAX_LAYER / (film.get_layer_number() * 2 + 1))
    if insert_search_pts <= 0:
        return False # unable to insert due to GPU limitation
    
    insert_idx_arr = make_test_insert_film(film, insert_search_pts)

    grad = get_insert_grad(film, target_spec_ls)
    
    greedy_insert_idx = np.argmin(grad[insert_idx_arr])
    greedy_insert_layer_idx = insert_idx_arr[greedy_insert_idx]

    # remove test layers but keep the best layer (needle insertion)
    film.remove_negative_thickness_layer(exclude=greedy_insert_layer_idx)
    return True



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
            film.calculate_n_inc(s.WLS)])
        
    # allocate space and calculate J and f
    d = film.get_d()
    J = np.empty((target_spec.shape[0], d.shape[0]))
    f = np.empty(target_spec.shape[0]) # only R spec: no absorption
    stack_J(J, n_arrs_ls, d, target_spec_ls, get_J=get_insert_jacobi_simple)
    stack_f(f, n_arrs_ls, d, target_spec_ls, target_spec)

    # find insertion place with largest negative gradient
    grad = np.dot(J.T, f)
    return grad



def make_test_insert_film(film, insert_search_pts):
    '''
    Add to every layer evenly insert_search_pts test insert layers,
    as a trail film so that insertion grad at those places could be 
    calculated

    '''
    # ensure not exceed cuda restrictions
    # which is MAX_LAYER layers (inserted) or layer * insert_pts * 2 < MAX_LAYER
    assert film.get_layer_number() * (insert_search_pts * 2 + 1) < MAX_LAYER, \
        'too many search points.'
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
