import sys
sys.path.append('./designer/script/')

import numpy as np
import copy

from film import FilmSimple
from spectrum import BaseSpectrum
from optimizer.LM_gradient_descent import stack_f, stack_J


def insert_1_layer(
    film: FilmSimple, 
    target_spec_ls: list[BaseSpectrum], 
    insert_search_pts=10, 
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
    
    insert_idx_arr = make_test_insert_film(film, insert_search_pts)

    # stack parameters & preparations
    d = film.get_d()
    target_spec = np.array([])
    n_arrs_ls = []
    for s in target_spec_ls:
        target_spec = np.append(target_spec, s.get_R())
        # calculate refractive indices in advance and store

        # In LM optimization this saves time but in needle
        # insertion it does not. Only to stay close to the 
        # implementation in LM descent for reusing code

        n_arrs_ls.append([
            film.calculate_n_array(s.WLS), 
            film.calculate_n_sub(s.WLS), 
            film.calculate_n_inc(s.WLS)])


    # allocate space and calculate J and f
    J = np.empty((target_spec.shape[0], d.shape[0]))
    f = np.empty(target_spec.shape[0]) # only R spec: no absorption
    stack_J(J, d, n_arrs_ls, target_spec_ls)
    stack_f(f, d, n_arrs_ls, target_spec_ls, target_spec)

    # find insertion place with largest negative gradient
    grad = np.dot(J.T, f)
    greedy_insert_idx = np.argmin(grad[insert_idx_arr])
    greedy_insert_layer_idx = insert_idx_arr[greedy_insert_idx]

    # remove test layers but keep the best layer (needle insertion)
    film.remove_negative_thickness_layer(exclude=greedy_insert_layer_idx)
    return 



def make_test_insert_film(film, insert_search_pts):
    '''
    Add to every layer evenly insert_search_pts test insert layers,
    as a trail film so that insertion grad at those places could be 
    calculated

    '''
    insert_idx_arr = [j * 2 + i * (2 * insert_search_pts + 1) 
                        for i in range(film.get_layer_number()) 
                        for j in range(insert_search_pts)]
    d_before = film.get_d().copy()
    for i in range(film.get_layer_number()):
        
        for j in range(insert_search_pts):
            insert_position = 1 / insert_search_pts * d_before[i]
            # insert new layer: zero thickness
            film.insert_layer(j * 2 + i * (2 * insert_search_pts + 1), insert_position, 0) 

    return insert_idx_arr
