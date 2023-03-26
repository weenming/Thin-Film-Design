import numpy as np
from gets.get_jacobi import get_jacobi_simple
from gets.get_spectrum import get_spectrum_simple

from film import FilmSimple
from spectrum import BaseSpectrum

def LM_optimize_d_simple(film: FilmSimple, target_spec_ls: list[BaseSpectrum], h_tol, max_step):
    """
    
    """
    layer_number = film.get_layer_number()
    
    target_spec = np.array([])
    n_arrs_ls = []
    for s in target_spec_ls:
        target_spec = np.append(target_spec, s.get_R())
        # calculate refractive indices in advance and store to save time
        n_arrs_ls.append([
            film.calculate_n_array(s.WLS), 
            film.calculate_n_sub(s.WLS), 
            film.calculate_n_inc(s.WLS)])

    # d: current d of designed film
    d = film.get_d()

    # allocate memory for J and f
    J = np.empty((target_spec.shape[0], d.shape[0]))
    f = np.empty(target_spec.shape[0]) # shape of reflectance spec: no absorption
    f_new = np.empty(target_spec.shape[0]) # shape of reflectance spec: no absorption

    # before first iteration, calculate g and A
    stack_J(J, d, n_arrs_ls, target_spec_ls) # use address reference, no return value
    stack_f(f, d, n_arrs_ls, target_spec_ls, target_spec)
    g = np.dot(J.T, f)
    A = np.dot(J.T, J)
    nu = 2
    mu = 1

    for step_count in range(max_step):
        stack_J(J, d, n_arrs_ls, target_spec_ls)
        stack_f(f, d, n_arrs_ls, target_spec_ls, target_spec)
        h = np.dot(np.linalg.inv(A + mu * np.identity(layer_number)), -g)
        d_new = d + h
        F_d = np.sum(np.square(f))
        
        # Strategy: do not allow negative thickness.
        # should not cause unexpected stopping because there should be other
        #  descending directions
        for i in range(d_new.shape[0]):
            if d_new[i] < 0: # project back to feasible domain
                d_new[i] = 0

        stack_f(f_new, d_new, n_arrs_ls, target_spec_ls, target_spec)
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

    film.update_d(d)
    film.check_thickness()
    film.remove_negative_thickness_layer()



def stack_f(
    f_old, 
    d, 
    n_arrs_ls: list[list[np.array]], 
    target_spec_ls: list[BaseSpectrum], 
    target_spec
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
        wls_num = s.WLS.shape[0]
        # note that numpy array slicing does not allocate new space in memory
        get_spectrum_simple(
            f_old[i: i + wls_num],
            s.WLS,
            d,
            n_arrs[0],
            n_arrs[1],
            n_arrs[2],
            s.INC_ANG
        )[:wls_num, :] # get only half of the spectrum
        i += wls_num
    
    f_old = f_old - target_spec
    return

def stack_J(
    J_old, 
    n_arrs_ls, 
    d, 
    target_spec_ls: list[BaseSpectrum]
):
    """
    target specs may have to be calculated using different params
    """
    i = 0
    for s, n_arrs in zip(target_spec_ls, n_arrs_ls):
        this_wls_num = s.WLS.shape[0]
        # only reflectance
        get_jacobi_simple(
            J_old[i: i + this_wls_num, :],
            s.WLS, 
            d,
            n_arrs[0],
            n_arrs[1], 
            n_arrs[2], 
            s.INC_ANG
        )[:s.WLS.shape[0], :]
        i += this_wls_num
    return

