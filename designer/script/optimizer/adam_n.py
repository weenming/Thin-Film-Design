import sys
sys.path.append('./designer/script/')


from tmm.get_jacobi_n_adjoint import get_jacobi_free_form
from tmm.get_spectrum import get_spectrum_free

from optimizer.grad_helper import stack_f, stack_J, stack_init_params
from utils.loss import calculate_RMS_f_spec
from spectrum import BaseSpectrum
from film import FreeFormFilm
import numpy as np
from typing import Sequence
import copy


def adam_optimize(
    film: FreeFormFilm,
    target_spec_ls: Sequence[BaseSpectrum],
    max_steps,
    alpha=0.001,  # learning rate TODO: fine tune this? but dependent
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    record=False,
    show=False,
    batch_size_spec=None,
    batch_size_wl=None,
    n_min=0,
    n_max=float('inf')
):
    '''
    Implements Adam gd. Only supports equal wl shuffle for now, which is that 
    every spectrum (inc / polarization) needs to have the same wavelength size.

    Here n is optimized where n is the non-dispersive refractive index of each
    layer.

    Adapted from Kingma, Diederik P. and Jimmy Ba.
     "Adam: A Method for Stochastic Optimization." CoRR abs/1412.6980 (2014)

    Parameters:
        batch_size_wl: 
            size of each wl mini-batch. for every wl, R and T are 
            both calculated. Therefore, J and f have size (in the 
            corresponding axis): batch_size_wl.shape[0] * 2.

    Returns:
        losses
        films
    '''

    # check batch size
    wl_num_min = np.min([s.WLS.shape[0] for s in target_spec_ls])
    if batch_size_spec is None:
        batch_size_spec = len(target_spec_ls)
    if batch_size_wl is None:
        batch_size_wl = wl_num_min
    assert batch_size_spec <= len(target_spec_ls) \
        and batch_size_wl <= wl_num_min  # spec with smallest wl

    # Prep: calculate refractive index & stack target spectrum into one array
    n_arrs_ls = stack_init_params(film, target_spec_ls)
    n = film.get_n()

    # avoid grad explode by asserting no total reflection
    n_min = film.calculate_n_inc(target_spec_ls[0].WLS)[0] * \
        np.sin(target_spec_ls[0].INC_ANG) if n_min == 0 else n_min

    total_wl_num = batch_size_wl * batch_size_spec * 2  # R & T
    J = np.empty((total_wl_num, n.shape[0]))
    f = np.empty(total_wl_num)

    if record:
        losses = [calculate_RMS_f_spec(film, target_spec_ls)]
        films = [copy.deepcopy(film)]

    # initialize
    m = 0
    v = 0

    for t in range(max_steps):
        # shuffle for sgd
        spec_batch_idx = np.random.default_rng().choice(
            len(target_spec_ls),
            batch_size_spec,
            replace=False
        )
        spec_batch_idx = np.sort(spec_batch_idx)
        # shuffle R and T.
        # mat: #wls \cross #spec; pick out elem on the crossing of
        # rows=wl_idx and cols=spec_idx. For selected wl, R and T
        # are calculated simultaneously.
        #
        # The size of J is fixed but
        # the stored grads are different in each epoch according to
        # the random shuffle.
        wl_batch_idx = np.random.default_rng().choice(
            wl_num_min,
            batch_size_wl,
            replace=False
        )
        wl_batch_idx = np.sort(wl_batch_idx)

        stack_J(
            J,
            n_arrs_ls,
            film.get_d(),
            target_spec_ls,
            MAX_LAYER_NUMBER=250,
            spec_batch_idx=spec_batch_idx,
            wl_batch_idx=wl_batch_idx,
            get_J=get_jacobi_free_form
        )
        stack_f(
            f,
            n_arrs_ls,
            film.get_d(),
            target_spec_ls,
            spec_batch_idx=spec_batch_idx,
            wl_batch_idx=wl_batch_idx,
            get_f=get_spectrum_free
        )

        g = J.T @ f
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)

        n = n - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Project back to feasible domain
        n[n < n_min] = n_min
        n[n > n_max] = n_max

        film.update_n(n)
        update_n(film, target_spec_ls, n_arrs_ls)

        # record and show functions
        if record:
            losses.append(calculate_RMS_f_spec(film, target_spec_ls))
            films.append(copy.deepcopy(film))
        if show:
            print(
                f'iter {t}, loss {calculate_RMS_f_spec(film, target_spec_ls)}')

        # if loss not decreasing, break
        try:
            if losses[-1] == losses[-2]:
                if np.array_equal(losses[-10:], [losses[-1]] * 10):
                    print('convergent, terminate eraly')
                    break
        except Exception as e:
            continue

    if record:
        return losses, films

    return None


def update_n(film: FreeFormFilm, target_spec_ls: Sequence[BaseSpectrum], n_arrs_ls):
    '''
    Parameters:
        n_arr_ls: [n_arr_1, n_arr_2, ...] (most outer list is different spectrums)
            n_arr_1: wls \times d
        n_arrs_ls: outdated n_arrs_ls, containing d and inc and sub
    '''
    for l, s in zip(n_arrs_ls, target_spec_ls):
        l[0] = film.calculate_n_array(s.WLS)
