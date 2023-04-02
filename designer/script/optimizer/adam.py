import numpy as np
import sys
sys.path.append('./designer/script/')

from film import FilmSimple
from spectrum import BaseSpectrum
from utils.loss import calculate_RMS_f_spec

from grad_helper import stack_f, stack_J


def adam_optimize(
        film: FilmSimple,
        target_spec_ls: list[BaseSpectrum], 
        max_steps, 
        alpha=0.001, # stepsize TODO: fine tune this...
        beta1 = 0.9, 
        beta2 = 0.999, 
        epsilon = 1e-8, 
        record = False, 
        show=False
    ):
    # Adapted from Kingma, Diederik P. and Jimmy Ba. 
    # "Adam: A Method for Stochastic Optimization." CoRR abs/1412.6980 (2014)

    # Prep: calculate refractive index & stack target spectrum into one array
    target_spec = np.array([])
    n_arrs_ls = []
    for s in target_spec_ls:
        # both R and T are calculated
        target_spec = np.append(target_spec, s.get_R())
        target_spec = np.append(target_spec, s.get_T())
        # calculate refractive indices in advance and store to save time
        n_arrs_ls.append([
            film.calculate_n_array(s.WLS), 
            film.calculate_n_sub(s.WLS), 
            film.calculate_n_inc(s.WLS)])
    d = film.get_d()
    J = np.empty((target_spec.shape[0], d.shape[0]))
    f = np.empty(target_spec.shape[0])

    losses = []

    # initialize
    m = 0
    v = 0
    
    for t in range(max_steps):
        stack_J(J, n_arrs_ls, d, target_spec_ls)
        stack_f(f, n_arrs_ls, d, target_spec_ls, target_spec)

        g = J.T @ f
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m  / (1 - beta1)
        v_hat = v / (1 - beta2)

        d = d - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Project back to feasible domain
        d[d < 0] = 0.

        if record:
            losses.append(calculate_RMS_f_spec(film, target_spec_ls))
        if show:
            print(f'iter {t}, loss {calculate_RMS_f_spec(film, target_spec_ls)}')
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


