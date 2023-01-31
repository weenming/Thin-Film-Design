import numpy as np
from gets.get_jacobi import get_jacobi
from gets.get_spectrum import get_spectrum
import time
from film import calculate_merit
from film import FilmSimple

def LM_optimize_d_simple(film: FilmSimple, target_film: FilmSimple, h_tol, max_step):
    """
    
    """
    layer_number = film.get_layer_number()
    target_spec_ls = target_film.get_all_spec_list()
    
    # target spectrum: concatenate target spectrums to a single 1d array
    target_spectrum = np.array([])
    # other spectrum parameters are organized into a list by different inc_angs
    wls_ls = []
    wls_number = 0
    inc_ang_ls = []
    # refractive indices also needs preparation
    n_layers_ls = []
    n_sub_ls = []
    n_inc_ls = []
    
    for s in target_spec_ls:
        # only reflectance spectrum
        target_spectrum = np.append(target_spectrum, s.get_R())
        wls_ls.append(s.WLS)
        wls_number += s.WLS.shape[0]
        inc_ang_ls.append(s.INC_ANG)
        n_layers_ls.append(film.calculate_n_array(s.WLS))
        n_sub_ls.append(film.calculate_n_sub(s.WLS))
        n_inc_ls.append(film.calculate_n_inc(s.WLS))

    # d: current d of designed film
    d = film.get_d()

    # TODO: allocate memory for J and f

    # before first iteration, calculate g and A
    J = stack_J(J, wls_ls, d, n_layers_ls, n_sub_ls, n_inc_ls, inc_ang_ls)
    f = stack_f(f, wls_ls, d, n_layers_ls, n_sub_ls, n_inc_ls, inc_ang_ls, 
                target_spectrum)
    g = np.dot(J.T, f)
    A = np.dot(J.T, J)
    nu = 2
    mu = 1

    for step_count in range(max_step):
        J = stack_J(J, wls_ls, d, n_layers_ls, n_sub_ls, n_inc_ls, inc_ang_ls)
        f = stack_f(f, wls_ls, d, n_layers_ls, n_sub_ls, n_inc_ls, inc_ang_ls, 
                    target_spectrum)
        h = np.dot(np.linalg.inv(A + mu * np.identity(layer_number)), -g)
        d_new = d + h
        F_d = np.sum(np.square(f))
        
        # Strategy: do not allow negative thickness.
        # should not cause unexpected stopping because there should be other
        #  descending directions
        for i in range(d_new.shape[0]):
            if d_new[i] < 0:
                d_new[i] = 0

        f_new = stack_f(f_new, wls_ls, d_new, n_layers_ls, n_sub_ls, n_inc_ls, 
                        inc_ang_ls, target_spectrum)
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



def stack_f(f_old, wls_num, layer_num, wls_ls, d, n_layers_ls, n_sub_ls, n_inc_ls, 
            inc_ang_ls, target_spec):
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
    for wls, inc_ang in zip(wls_ls, inc_ang_ls):
        this_wls_num = wls.shape[0]
        f_old[i: i + this_wls_num] = get_spectrum(
            wls,
            d,
            n_layers_ls[-1],
            n_sub_ls[-1], 
            n_inc_ls[-1], 
            inc_ang[-1]
        )[:wls.shape[0], :]
        i += this_wls_num
    
    f_old = f_old - target_spec
    return

def stack_J(J_old, wls_num, layer_num, wls_ls, d, n_layers_ls, n_sub_ls, n_inc_ls, 
            inc_ang_ls):
    """
    target specs may have to be calculated using different params
    """
    i = 0
    for wls, inc_ang in zip(wls_ls, inc_ang_ls):
        this_wls_num = wls.shape[0]
        # only reflectance
        J_old[i: i + this_wls_num, :] = get_jacobi(
            wls, 
            d,
            n_layers_ls[-1],
            n_sub_ls[-1], 
            n_inc_ls[-1], 
            inc_ang[-1]
        )[:wls.shape[0], :]
        i += this_wls_num
        
    return

