import numpy as np
from tmm.get_jacobi import get_jacobi_simple
from tmm.get_spectrum import get_spectrum_simple

from film import TwoMaterialFilm
from spectrum import BaseSpectrum, Spectrum


def stack_init_params(film: TwoMaterialFilm, target_spec_ls: list[BaseSpectrum]):
    # stack parameters & preparations
    target_spec = np.array([])
    n_arrs_ls = []
    for s in target_spec_ls:
        # calculate refractive indices in advance and store

        # In LM optimization this saves time but in needle
        # insertion it does not. Only to stay close to the
        # implementation in LM descent for reusing code

        n_arrs_ls.append([
            film.calculate_n_array(s.WLS),
            film.calculate_n_sub(s.WLS),
            film.calculate_n_inc(s.WLS)
        ])
    return n_arrs_ls


def stack_f(
    f_old,
    n_arrs_ls: list[list[np.typing.NDArray]],
    d,
    target_spec_ls: list[BaseSpectrum],
    get_f=get_spectrum_simple,
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
        wls_num = s.WLS.shape[0]  # R and T
        # note that numpy array slicing does not allocate new space in memory
        get_f(
            f_old[wl_idx: wl_idx + wls_num * 2],
            s.WLS,
            d,
            n_arrs[0],
            n_arrs[1],  # n_sub
            n_arrs[2],  # n_inc
            s.INC_ANG
        )
        # should not create new arr.
        f_old[wl_idx: wl_idx + wls_num] -= s.get_R()
        f_old[wl_idx + wls_num: wl_idx + 2 * wls_num] -= s.get_T()

        wl_idx += wls_num * 2
    return


def stack_J(
    J_old,
    n_arrs_ls,
    d: np.typing.NDArray,
    target_spec_ls: list[BaseSpectrum],
    get_J=get_jacobi_simple,
    MAX_LAYER_NUMBER=250,
):
    """
    target specs may have to be calculated using different params.
    Note that calculation of Jacobian consumes a memory that scales
    with layer number. When too large, must split up.
    """
    wl_idx = 0
    for s, n_arrs in zip(target_spec_ls, n_arrs_ls):
        wl_num = s.WLS.shape[0]  # R and T
        # only reflectance
        get_J(
            J_old[wl_idx: wl_idx + wl_num * 2, :],
            s.WLS,
            d[:],
            n_arrs[0][:, :],
            n_arrs[1][:],  # n_sub
            n_arrs[2][:],  # n_inc
            s.INC_ANG,
        )
        wl_idx += wl_num * 2
    return
