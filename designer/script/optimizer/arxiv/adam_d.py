import sys
sys.path.append('./designer/script/')


from tmm.get_jacobi import get_jacobi_simple
from tmm.get_spectrum import get_spectrum_simple
from optimizer.grad_helper import stack_f, stack_J, stack_init_params
from utils.loss import calculate_RMS_f_spec, rms
from spectrum import BaseSpectrum
from film import TwoMaterialFilm
import numpy as np
from typing import Sequence
import copy


def adam_optimize(
    film: TwoMaterialFilm,
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
):
    '''
    Implements Adam gd. Only supports equal wl shuffle for now, which is that 
    every spectrum (inc / polarization) needs to have the same wavelength size.

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
    d = film.get_d()
    total_wl_num = batch_size_wl * batch_size_spec * 2  # R & T
    J = np.empty((total_wl_num, d.shape[0]))
    f = np.empty(total_wl_num)

    losses = []
    films = []

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
            get_J=get_jacobi_simple
        )
        stack_f(
            f,
            n_arrs_ls,
            film.get_d(),
            target_spec_ls,
            spec_batch_idx=spec_batch_idx,
            wl_batch_idx=wl_batch_idx,
            get_f=get_spectrum_simple
        )

        g = J.T @ f
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)

        d = d - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Project back to feasible domain
        d[d < 0] = 0.

        film.update_d(d)

        if record:
            losses.append(rms(f))
            films.append(copy.deepcopy(film))
        if show:
            print(
                f'iter {t}, loss {rms(f)}')

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
