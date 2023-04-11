from optimizer.grad_helper import stack_f, stack_J, stack_init_params
from utils.loss import calculate_RMS_f_spec
from spectrum import BaseSpectrum
from film import FilmSimple
import numpy as np
import sys
sys.path.append('./designer/script/')


def adam_optimize(
    film: FilmSimple,
    target_spec_ls: list[BaseSpectrum],
    max_steps,
    alpha=0.001,  # learning rate TODO: fine tune this? but dependent
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    record=False,
    show=False,
    batch_size_spec=None,
    batch_size_wl=None
):
    # Adapted from Kingma, Diederik P. and Jimmy Ba.
    # "Adam: A Method for Stochastic Optimization." CoRR abs/1412.6980 (2014)

    # check batch size
    assert batch_size_spec <= len(target_spec_ls) \
        and batch_size_wl < np.min([s.WLS.shape[0] for s in target_spec_ls])

    # Prep: calculate refractive index & stack target spectrum into one array
    target_spec, n_arrs_ls = stack_init_params(film, target_spec_ls)
    d = film.get_d()
    J = np.empty((target_spec.shape[0], d.shape[0]))
    f = np.empty(target_spec.shape[0])

    losses = []

    # initialize
    m = 0
    v = 0

    for t in range(max_steps):
        stack_J(J, n_arrs_ls, d, target_spec_ls, MAX_LAYER_NUMBER=250)
        stack_f(f, n_arrs_ls, d, target_spec_ls)

        g = J.T @ f
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)

        d = d - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Project back to feasible domain
        d[d < 0] = 0.

        if record:
            losses.append(calculate_RMS_f_spec(film, target_spec_ls))
        if show:
            print(
                f'iter {t}, loss {calculate_RMS_f_spec(film, target_spec_ls)}')
        film.update_d(d)

        # if loss not decreasing, break
        try:
            if losses[-1] == losses[-2]:
                if np.array_equal(losses[-10:], [losses[-1]] * 10):
                    print('convergent, terminate eraly')
                    break
        except Exception as e:
            continue

    if record:
        return losses

    return None
