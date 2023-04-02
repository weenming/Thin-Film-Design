
import numpy as np
from gets.get_jacobi import get_jacobi_simple
from gets.get_spectrum import get_spectrum_simple

from film import FilmSimple
from spectrum import BaseSpectrum

def stack_f(
    f_old, 
    n_arrs_ls: list[list[np.array]], 
    d, 
    target_spec_ls: list[BaseSpectrum], 
    target_spec,
    get_f = get_spectrum_simple, 
):
    """
    target specs may have to be calculated using different params

    This implementation uses dynamic sized list but they should 
    have acceptable performance because list elems are refrerences
    to numpy arrays

    Note that calculating spec does not have a limitation on the layer number. 

    Arguments:
        f_old:
            Old postion of f. Its size does not change in a LM optimization so the 
            memory only needs to be allocated once at initialization.
        wls_num (int):
            sum of number of wl points in the wls_ls
        layer_num (int):
            layer number
    """
    wl_idx = 0
    for s, n_arrs in zip(target_spec_ls, n_arrs_ls):
        wls_num = s.WLS.shape[0] * 2 # R and T
        # note that numpy array slicing does not allocate new space in memory
        get_f(
            f_old[wl_idx: wl_idx + wls_num],
            s.WLS,
            d,
            n_arrs[0],
            n_arrs[1],
            n_arrs[2],
            s.INC_ANG
        )
        wl_idx += wls_num
    
    f_old[:] = f_old - target_spec # should not create new arr.
    return

def stack_J(
    J_old, 
    n_arrs_ls, 
    d, 
    target_spec_ls: list[BaseSpectrum],
    get_J = get_jacobi_simple, 
    MAX_LAYER_NUMBER=250, 
):
    """
    target specs may have to be calculated using different params.

    Note that calculation of Jacobian consumes a memory that scales
    with layer number. When too large, must split up.
    """
    wl_idx = 0
    d_idx = 0
    M = MAX_LAYER_NUMBER
    for layer_idx in range((d.shape[0] - 1) // M + 1):
        for s, n_arrs in zip(target_spec_ls, n_arrs_ls):
            wl_num = s.WLS.shape[0] * 2 # R and T
            # only reflectance
            get_J(
                J_old[wl_idx: wl_idx + wl_num, :],
                s.WLS, 
                d,
                n_arrs[0][:, d_idx: d_idx + M],
                n_arrs[1], 
                n_arrs[2], 
                s.INC_ANG
            )
            wl_idx += wl_num
    return



    '''
            [[
            y[0][:, i * M: (i + 1) * M], 
            y[1][i * M: (i + 1) * M], 
            y[2][i * M: (i + 1) * M]
        ] for y in n_arrs_ls], # i know it is shit :( '''