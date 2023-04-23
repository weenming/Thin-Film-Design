
import numpy as np
from tmm.get_jacobi import get_jacobi_simple
from tmm.get_spectrum import get_spectrum_simple
from typing import Sequence
from film import TwoMaterialFilm, BaseFilm
from spectrum import BaseSpectrum, Spectrum


def stack_init_params(
    film: BaseFilm,
    target_spec_ls: Sequence[BaseSpectrum],
):
    # stack parameters & preparations
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
    n_arrs_ls: Sequence[Sequence[np.typing.NDArray]],
    d,
    target_spec_ls: Sequence[BaseSpectrum],
    spec_batch_idx=None,
    wl_batch_idx=None,
    get_f=get_spectrum_simple,
):
    """
    Calculates f  w.r.t a list objective spectrums and add them together.

    Target specs may have to be calculated using different params

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
    if spec_batch_idx is None:
        spec_batch_idx = list(range(len(target_spec_ls)))
    if wl_batch_idx is None:
        wl_num_min = np.min([s.WLS.shape[0] for s in target_spec_ls])
        wl_batch_idx = np.arange(wl_num_min)

    wl_idx = 0
    for i, (s, n_arrs) in enumerate(zip(target_spec_ls, n_arrs_ls)):

        wl_num = wl_batch_idx.shape[0]

        if i not in spec_batch_idx:  # for SGD
            continue

        # note that numpy array slicing does not allocate new space in memory
        get_f(
            f_old[wl_idx: wl_idx + wl_num * 2],  # R & T
            s.WLS[wl_batch_idx],
            d,
            n_arrs[0][wl_batch_idx, :],
            n_arrs[1][wl_batch_idx],  # n_sub
            n_arrs[2][wl_batch_idx],  # n_inc
            s.INC_ANG
        )

        # should not create new arr.
        f_old[wl_idx: wl_idx + wl_num] -= s.get_R()[wl_batch_idx]
        f_old[wl_idx + wl_num: wl_idx + wl_num * 2] -= s.get_T()[wl_batch_idx]

        wl_idx += wl_num * 2
    return


def stack_J(
    J_old,
    n_arrs_ls,
    d: np.typing.NDArray,
    target_spec_ls: Sequence[BaseSpectrum],
    get_J=get_jacobi_simple,
    MAX_LAYER_NUMBER=250,
    spec_batch_idx=None,
    wl_batch_idx=None,
):
    """
    Calculates J  w.r.t a list objective spectrums and add them together.

    target specs may have to be calculated using different params.

    Note that calculation of Jacobian consumes a memory that scales
    with layer number. When too large, must split up.
    """
    if spec_batch_idx is None:
        spec_batch_idx = list(range(len(target_spec_ls)))
    if wl_batch_idx is None:
        wl_num_min = np.min([s.WLS.shape[0] for s in target_spec_ls])
        wl_batch_idx = np.arange(wl_num_min)

    wl_count = 0
    for i, (s, n_arrs) in enumerate(zip(target_spec_ls, n_arrs_ls)):

        wl_num = wl_batch_idx.shape[0]  # R and T: batch can be 2 #wl long

        if i not in spec_batch_idx:  # mini-batching
            continue

        get_J(
            J_old[wl_count: wl_count + wl_num * 2, :],  # R & T
            s.WLS[wl_batch_idx],
            d[:],
            n_arrs[0][wl_batch_idx, :],
            n_arrs[1][wl_batch_idx],  # n_sub
            n_arrs[2][wl_batch_idx],  # n_inc
            s.INC_ANG,
        )
        wl_count += wl_num * 2
    return
