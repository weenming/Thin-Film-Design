import numpy as np
from gets.get_jacobi import get_jacobi_simple
from gets.get_spectrum import get_spectrum_simple

from film import FilmSimple
from spectrum import BaseSpectrum

from optimizer.grad_helper import stack_f, stack_J, stack_init_params


def LM_optimize_d_simple(
        film: FilmSimple, 
        target_spec_ls: list[BaseSpectrum], 
        h_tol, 
        max_step, 
        show=False, 
        record = False
):
    """
    
    """

    # Prep: calculate refractive index & stack target spectrum into one array
    target_spec, n_arrs_ls = stack_init_params(film, target_spec_ls)
    d = film.get_d()

    # check layer number. 
    
    # allocate memory for J and f
    J = np.empty((target_spec.shape[0], d.shape[0]))
    f = np.empty(target_spec.shape[0])
    f_new = np.empty(target_spec.shape[0])

    losses = []

    # Initialize for LM: before first iteration, calculate g and A
    stack_J(J, n_arrs_ls, d, target_spec_ls) # adr ref, no ret val
    stack_f(f, n_arrs_ls, d, target_spec_ls, target_spec)
    g = np.dot(J.T, f)
    A = np.dot(J.T, J)
    nu = 2
    mu = 1

    for step_count in range(max_step):

        h = np.dot(np.linalg.inv(A + mu * np.identity(d.shape[0])), -g)
        d_new = d + h
        F_d = np.sum(np.square(f))
        
        # Strategy: project back to feasible domain: d_i > 0
        # should not cause unexpected stopping because there should be other
        # descending directions
        for i in range(d_new.shape[0]):
            if d_new[i] < 0:
                d_new[i] = 0

        stack_f(f_new, n_arrs_ls, d_new, target_spec_ls, target_spec)
        F_dnew = np.sum(np.square(f_new))
        rho = (F_d - F_dnew) / np.dot(h.T, mu * h - g).item()

        # Accept this move by merit
        if rho > 0:
            d = d_new.copy()
            # Here f points to the space of f_new and the old the reference count 
            # of f -=1.
            # Thus to keep the space of f from being recycled, a tmp reference is 
            # added. 
            tmp = f
            f = f_new
            f_new = tmp
            # After the "swap", f_new now points to the location of where f was. 
            # It does not matter that f_new now has dirty data becaus f_new will
            # be updated before next time F_new is calculated based on it.
            stack_J(J, n_arrs_ls, d, target_spec_ls)
            stack_f(f, n_arrs_ls, d, target_spec_ls, target_spec)
            g = np.dot(J.T, f)
            A = np.dot(J.T, J)
            mu = mu * \
                np.amax(np.array([1 / 3, 1 - (2 * rho - 1) ** 3])).item()
            nu = 2
        else:
            mu = mu * nu
            nu = 2 * nu
        if np.max(np.abs(h)).item() < h_tol:
            break
        
        if show:
            loss = np.sqrt(F_d / f.shape[0])
            print(f'loss: {loss}')
        if record:
            loss = np.sqrt(F_d / f.shape[0])
            losses.append(loss)

    film.update_d(d)
    film.remove_negative_thickness_layer()
    if film.get_layer_number() == 0:
        raise Exception('Design terminated: zero layers')
    if record:
        return step_count, losses
    return step_count




