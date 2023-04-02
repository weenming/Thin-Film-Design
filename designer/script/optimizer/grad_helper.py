
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
    get_f = get_spectrum_simple
):
    """
    target specs may have to be calculated using different params

    This implementation uses dynamic sized list...

    Arguments:
        f_old:
            Old postion of f. Its size does not change in a LM optimization so the 
            memory only needs to be allocated once at initialization.
        wls_num (int):
            sum of number of wl points in the wls_ls
        layer_num (int):
            layer number
    """
    i = 0
    for s, n_arrs in zip(target_spec_ls, n_arrs_ls):
        wls_num = s.WLS.shape[0] * 2 # R and T
        # note that numpy array slicing does not allocate new space in memory
        get_f(
            f_old[i: i + wls_num],
            s.WLS,
            d,
            n_arrs[0],
            n_arrs[1],
            n_arrs[2],
            s.INC_ANG
        )
        i += wls_num
    
    f_old[:] = f_old - target_spec # should not create new arr.
    return

def stack_J(
    J_old, 
    n_arrs_ls, 
    d, 
    target_spec_ls: list[BaseSpectrum],
    get_J = get_jacobi_simple
):
    """
    target specs may have to be calculated using different params
    """
    i = 0
    for s, n_arrs in zip(target_spec_ls, n_arrs_ls):
        this_wls_num = s.WLS.shape[0] * 2 # R and T
        # only reflectance
        get_J(
            J_old[i: i + this_wls_num, :],
            s.WLS, 
            d,
            n_arrs[0],
            n_arrs[1], 
            n_arrs[2], 
            s.INC_ANG
        )
        i += this_wls_num
    return