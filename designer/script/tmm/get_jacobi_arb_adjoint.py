import numpy as np
import cmath
from numba import cuda
from tmm.mat_utils import calc_M, calc_M_inv, fill_arr, mul_to, mul_right, mul_left, hadm_mul  # multiply
from tmm.mat_utils import tsp  # transpose


def get_jacobi_E_free_form(
    jacobi,
    wls,
    d,
    n_layers,
    n_sub,
    n_inc,
    inc_ang,
):
    """
    This function calculates the Jacobi matrix of a given TFNN. Back 
    propagation is implemented to acquire accurate result.
    The Jacobian is defined as: 
    $J_{imn} \coloneqq \partial E^{(i)}_{kl} / \partial M^{(j)}_{mn}$, 
    where i specifies wl
    mn denotes the element index of M.
    NOTE: I am not sure if TMM is analytic, but let's do this first
    NOTE: the return Jacobian shape is different from other functions. pay attention.s

    Parameters:
        jacobi (4D np.array):
            size: 4 wls.shape[0] \cross d.shape[0] \cross 2 \cross 2
            Pre-allocated memory space for returning Jacobian.
            Arranged as follows: (J_{E_+,s}, J_{E_+,p}, J_{E_s}, J_{E_+,p})
        wls (1d np.array):
            wavelengths of the target spectrum
        d (1d np.array):
            multi-layer thicknesses after last iteration
        n_layers (2d np.array): 
            size: wls.shape[0] \cross d.shape[0]. refractive indices of 
            each *layer*
        n_sub (1d np.array):
            refractive indices of the substrate
        n_inc (1d np.array):
            refractive indices of the incident material
        inc_ang (float):
            incident angle in degree
        s_ratio (float):
            portion of s-polarized light. Only intensity is taken into account,
            which means randomized phase difference is assumed.
        p_ratio (float):
            p-polarized light
    """
    # layer number of thin film, substrate not included
    layer_number = d.shape[0]
    # convert incident angle in degree to rad
    inc_ang_rad = inc_ang / 180 * np.pi
    # traverse all wl, save R and T to the 2N*1 np.array spectrum. [R, T]
    wls_size = wls.shape[0]

    # TODO: move the copy of wls, n arr to outer loop
    # (caller of spec, for example LM optimizer)
    # Maybe allowing it to pass in additional device arr would be a good idea

    # copy wls, d, n_layers, n_sub, n_inc, inc_ang to GPU
    wls_device = cuda.to_device(wls)
    d_device = cuda.to_device(d)
    n_layers_device = cuda.to_device(n_layers)
    n_sub_device = cuda.to_device(n_sub)
    n_inc_device = cuda.to_device(n_inc)

    # allocate space for Jacobi matrix
    jacobi_device = cuda.device_array(
        (wls_size * 4, layer_number, 2, 2),
        dtype="complex128"
    )

    # invoke kernel
    block_size = 16  # threads per block
    grid_size = (wls_size + block_size - 1) // block_size  # blocks per grid

    forward_and_backward_propagation[grid_size, block_size](
        jacobi_device,
        wls_device,
        d_device,
        n_layers_device,
        n_sub_device,
        n_inc_device,
        inc_ang_rad,
        wls_size,
        layer_number,
    )
    cuda.synchronize()
    # copy to pre-allocated space
    jacobi_device.copy_to_host(jacobi)


@cuda.jit
def forward_and_backward_propagation(
    jacobi,
    wls,
    d,
    n_layers,
    n_sub_arr,
    n_inc_arr,
    inc_ang,
    wls_size,
    layer_number,
):
    """
    Parameters:
        jacobi (cuda.device_array):
            size: wls_size * 2 \corss layer_number
            device array for storing calculated jacobi matrix
        wls (cuda.device_array):
            wavelengths
        d (cuda.device_array):
        n_A (cuda.device_array):
            n of material A at different wls
        n_B (cuda.device_array)
        n_sub
        n_inc
        inc_ang (float):
            incident angle in rad
        wls_size:
            number of wavelengths
        layer_number:
            number of layers
    """
    thread_id = cuda.grid(1)
    # check this thread is valid
    if thread_id > wls_size - 1:
        return
    # each thread calculates one wl
    wl = wls[thread_id]
    # inc_ang is already in rad
    n_arr = n_layers[thread_id, :]
    n_sub = n_sub_arr[thread_id]
    n_inc = n_inc_arr[thread_id]
    # Incident angle in each layer.
    # Snell's law: n_a sin(phi_a) = n_b sin(phi_b)
    cos_inc = cmath.cos(inc_ang)
    cos_sub = cmath.sqrt(1 - ((n_inc / n_sub) * cmath.sin(inc_ang)) ** 2)

    # calculate n_arr in real time

    '''
    FORWARD PROPAGATION
    '''

    # E_in = W_front M_i W_back E_out.

    W_back_s = cuda.local.array((2, 2), dtype="complex128")
    W_back_p = cuda.local.array((2, 2), dtype="complex128")

    fill_arr(W_back_s, 0.5, 0.5 / cos_inc / n_inc, 0.5, -0.5 / cos_inc / n_inc)
    fill_arr(W_back_p, 0.5 / n_inc, 0.5 / cos_inc, 0.5 / n_inc, -0.5 / cos_inc)

    Ms = cuda.local.array((2, 2), dtype="complex128")
    Mp = cuda.local.array((2, 2), dtype="complex128")

    for i in range(layer_number):

        calc_M(Ms, Mp, n_inc, inc_ang, n_arr[i], d[i], wl)
        mul_right(W_back_s, Ms)
        mul_right(W_back_p, Mp)

    # construct the last term D_{n+1}
    # technically this is merely D which is not M (DPD^{-1})
    fill_arr(Ms, 1, 1, n_sub * cos_sub, -n_sub * cos_sub)
    fill_arr(Mp, n_sub, n_sub, cos_sub, -cos_sub)
    mul_right(W_back_s, Ms)
    mul_right(W_back_p, Mp)

    # retrieve R and T (calculate the factor before energy flux)
    # Note that spectrum is array on device
    # Es_pos = W_back_s[0, 0]
    # Es_neg = W_back_s[1, 0]
    # Ep_pos = W_back_p[0, 0]
    # Ep_neg = W_back_p[1, 0]

    '''
    BACKWARD PROPAGATION
    '''
    partial_Es_pos = cuda.local.array((2, 2), dtype="complex128")
    partial_Es_neg = cuda.local.array((2, 2), dtype="complex128")
    partial_Ep_pos = cuda.local.array((2, 2), dtype="complex128")
    partial_Ep_neg = cuda.local.array((2, 2), dtype="complex128")

    # \partial_{W_{tot}} E_+
    fill_arr(partial_Es_pos, 1, 0, 0, 0)
    fill_arr(partial_Ep_pos, 1, 0, 0, 0)
    # \partial_{W_{tot}} T = t^* \partial_{W_{tot}} t
    fill_arr(partial_Es_neg, 0, 0, 1, 0)
    fill_arr(partial_Ep_neg, 0, 0, 1, 0)

    # partial TMM / partial M_ij
    partial_M00 = cuda.local.array((2, 2), dtype="complex128")
    partial_M01 = cuda.local.array((2, 2), dtype="complex128")
    partial_M10 = cuda.local.array((2, 2), dtype="complex128")
    partial_M11 = cuda.local.array((2, 2), dtype="complex128")
    fill_arr(partial_M00, 1, 0, 0, 0)
    fill_arr(partial_M01, 0, 1, 0, 0)
    fill_arr(partial_M10, 0, 0, 1, 0)
    fill_arr(partial_M11, 0, 0, 0, 1)

    W_front_s = cuda.local.array((2, 2), dtype="complex128")
    W_front_p = cuda.local.array((2, 2), dtype="complex128")
    Ms_inv = cuda.local.array((2, 2), dtype='complex128')
    Mp_inv = cuda.local.array((2, 2), dtype='complex128')
    tmp_res_s = cuda.local.array((2, 2), dtype='complex128')
    tmp_res_p = cuda.local.array((2, 2), dtype='complex128')

    # make front matrix
    fill_arr(W_front_s, 0.5, 0.5 / cos_inc /
             n_inc, 0.5, -0.5 / cos_inc / n_inc)
    fill_arr(W_front_p, 0.5 / n_inc, 0.5 /
             cos_inc, 0.5 / n_inc, -0.5 / cos_inc)

    # make back matrix
    fill_arr(Ms_inv, 1, 1, n_inc * cos_inc, -n_inc * cos_inc)
    fill_arr(Mp_inv, n_inc, n_inc, cos_inc, -cos_inc)
    mul_left(Ms_inv, W_back_s)  # D_0^-1 to left
    mul_left(Mp_inv, W_back_p)

    # special case: first layer
    calc_M_inv(Ms_inv, Mp_inv, n_inc, inc_ang, n_arr[0], d[0], wl)
    mul_left(Ms_inv, W_back_s)  # M_0^-1 to left
    mul_left(Mp_inv, W_back_p)  # M_0^-1 to left

    for i in range(layer_number):
        # M[i + 1] corresponds to i-th layer
        # (first layer with material A is the 0-th layer)

        # TODO: this is so stupid... no inline in python
        # horrible code...

        mul_to(W_front_s, partial_M00, tmp_res_s)
        mul_to(tmp_res_s, W_back_s, tmp_res_s)
        mul_to(W_front_p, partial_M00, tmp_res_p)
        mul_to(tmp_res_p, W_back_p, tmp_res_p)
        partial_s_pos = hadm_mul(tmp_res_s, partial_Es_pos)
        partial_p_pos = hadm_mul(tmp_res_p, partial_Ep_pos)
        partial_s_neg = hadm_mul(tmp_res_s, partial_Es_neg)
        partial_p_neg = hadm_mul(tmp_res_p, partial_Ep_neg)
        jacobi[thread_id + 0 * wls_size, i, 0, 0] = partial_s_pos
        jacobi[thread_id + 1 * wls_size, i, 0, 0] = partial_p_pos
        jacobi[thread_id + 2 * wls_size, i, 0, 0] = partial_s_neg
        jacobi[thread_id + 3 * wls_size, i, 0, 0] = partial_p_neg

        mul_to(W_front_s, partial_M01, tmp_res_s)
        mul_to(tmp_res_s, W_back_s, tmp_res_s)
        mul_to(W_front_p, partial_M01, tmp_res_p)
        mul_to(tmp_res_p, W_back_p, tmp_res_p)
        partial_s_pos = hadm_mul(tmp_res_s, partial_Es_pos)
        partial_p_pos = hadm_mul(tmp_res_p, partial_Ep_pos)
        partial_s_neg = hadm_mul(tmp_res_s, partial_Es_neg)
        partial_p_neg = hadm_mul(tmp_res_p, partial_Ep_neg)
        jacobi[thread_id + 0 * wls_size, i, 0, 1] = partial_s_pos
        jacobi[thread_id + 1 * wls_size, i, 0, 1] = partial_p_pos
        jacobi[thread_id + 2 * wls_size, i, 0, 1] = partial_s_neg
        jacobi[thread_id + 3 * wls_size, i, 0, 1] = partial_p_neg

        mul_to(W_front_s, partial_M10, tmp_res_s)
        mul_to(tmp_res_s, W_back_s, tmp_res_s)
        mul_to(W_front_p, partial_M10, tmp_res_p)
        mul_to(tmp_res_p, W_back_p, tmp_res_p)
        partial_s_pos = hadm_mul(tmp_res_s, partial_Es_pos)
        partial_p_pos = hadm_mul(tmp_res_p, partial_Ep_pos)
        partial_s_neg = hadm_mul(tmp_res_s, partial_Es_neg)
        partial_p_neg = hadm_mul(tmp_res_p, partial_Ep_neg)
        jacobi[thread_id + 0 * wls_size, i, 1, 0] = partial_s_pos
        jacobi[thread_id + 1 * wls_size, i, 1, 0] = partial_p_pos
        jacobi[thread_id + 2 * wls_size, i, 1, 0] = partial_s_neg
        jacobi[thread_id + 3 * wls_size, i, 1, 0] = partial_p_neg

        mul_to(W_front_s, partial_M11, tmp_res_s)
        mul_to(tmp_res_s, W_back_s, tmp_res_s)
        mul_to(W_front_p, partial_M11, tmp_res_p)
        mul_to(tmp_res_p, W_back_p, tmp_res_p)
        partial_s_pos = hadm_mul(tmp_res_s, partial_Es_pos)
        partial_p_pos = hadm_mul(tmp_res_p, partial_Ep_pos)
        partial_s_neg = hadm_mul(tmp_res_s, partial_Es_neg)
        partial_p_neg = hadm_mul(tmp_res_p, partial_Ep_neg)
        jacobi[thread_id + 0 * wls_size, i, 1, 1] = partial_s_pos
        jacobi[thread_id + 1 * wls_size, i, 1, 1] = partial_p_pos
        jacobi[thread_id + 2 * wls_size, i, 1, 1] = partial_s_neg
        jacobi[thread_id + 3 * wls_size, i, 1, 1] = partial_p_neg
    

        # update W_back and W_front
        if i == layer_number - 1:
            continue
        calc_M_inv(Ms_inv, Mp_inv, n_inc, inc_ang,
                   n_arr[i + 1], d[i + 1], wl)
        mul_left(Ms_inv, W_back_s)  # M_0^-1 to left
        mul_left(Mp_inv, W_back_p)  # M_0^-1 to left

        calc_M(Ms, Mp, n_inc, inc_ang, n_arr[i], d[i], wl)
        mul_right(W_front_s, Ms)  # M_0^-1 to left
        mul_right(W_front_p, Mp)  # M_0^-1 to left

 