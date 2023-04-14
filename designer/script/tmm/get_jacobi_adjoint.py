import numpy as np
import cmath
from numba import cuda
from tmm.mat_lib import mul_to, mul_right, mul_left, hadm_mul  # multiply
from tmm.mat_lib import tsp  # transpose


def get_jacobi_simple(
    jacobi,
    wls,
    d,
    n_layers,
    n_sub,
    n_inc,
    inc_ang,
    s_ratio=1,
    p_ratio=1
):
    """
    This function calculates the Jacobi matrix of a given TFNN. Back 
    propagation is implemented to acquire accurate result.
    NOTE: n_inc is not yet implemented

    Parameters:
        jacobi (2d np.array):
            size: wls.shape[0] \cross d.shape[0] 
            pre-allocated memory space for returning jacobi
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
    # copy 2d arr into 1d as contiguous arr to save data transfer
    n_A = n_layers[:, 0].copy()
    n_A_device = cuda.to_device(n_A)
    # may have only 1 layer.
    if layer_number == 1:
        n_B_device = cuda.to_device(np.empty(wls_size, dtype='complex128'))
    else:
        n_B = n_layers[:, 1].copy()
        n_B_device = cuda.to_device(n_B)
    n_sub_device = cuda.to_device(n_sub)
    n_inc_device = cuda.to_device(n_inc)

    # allocate space for Jacobi matrix
    jacobi_device = cuda.device_array(
        (wls_size * 2, layer_number),
        dtype="float64"
    )

    # invoke kernel
    block_size = 16  # threads per block
    grid_size = (wls_size + block_size - 1) // block_size  # blocks per grid

    forward_and_backward_propagation[grid_size, block_size](
        jacobi_device,
        wls_device,
        d_device,
        n_A_device,
        n_B_device,
        n_sub_device,
        n_inc_device,
        inc_ang_rad,
        wls_size,
        layer_number,
        s_ratio,
        p_ratio
    )
    cuda.synchronize()
    # copy to pre-allocated space
    jacobi_device.copy_to_host(jacobi)


@cuda.jit
def forward_and_backward_propagation(
    jacobi,
    wls,
    d,
    n_A_arr,
    n_B_arr,
    n_sub_arr,
    n_inc_arr,
    inc_ang,
    wls_size,
    layer_number,
    s_ratio,
    p_ratio
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
    n_A = n_A_arr[thread_id]
    n_B = n_B_arr[thread_id]
    n_sub = n_sub_arr[thread_id]
    n_inc = n_inc_arr[thread_id]
    # Incident angle in each layer.
    # Snell's law: n_a sin(phi_a) = n_b sin(phi_b)
    cos_A = cmath.sqrt(1 - ((n_inc / n_A) * cmath.sin(inc_ang)) ** 2)
    cos_B = cmath.sqrt(1 - ((n_inc / n_B) * cmath.sin(inc_ang)) ** 2)
    cos_inc = cmath.cos(inc_ang)
    cos_sub = cmath.sqrt(1 - ((n_inc / n_sub) * cmath.sin(inc_ang)) ** 2)

    # choose cos from arr of size 2.
    # Use local array which is private to thread
    cos_arr = cuda.local.array(2, dtype="complex128")
    cos_arr[0] = cos_A
    cos_arr[1] = cos_B

    n_arr = cuda.local.array(2, dtype="complex128")
    n_arr[0] = n_A
    n_arr[1] = n_B

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
        calc_M(Ms, Mp, cos_arr[i % 2], n_arr[i % 2], d[i], wl)
        mul_right(W_back_s, Ms)
        mul_right(W_back_p, Mp)

    # construct the last term D_{n+1}
    # technically this is merely D which is not M (D^{-2}PD)
    fill_arr(Ms, 1, 1, n_sub * cos_sub, n_sub * cos_sub)
    fill_arr(Mp, n_sub, n_sub, cos_sub, cos_sub)
    mul_right(W_back_s, Ms)
    mul_right(W_back_p, Mp)

    # retrieve R and T (calculate the factor before energy flux)
    # Note that spectrum is array on device
    rs = W_back_s[1, 0] / W_back_s[0, 0]
    rp = W_back_p[1, 0] / W_back_p[0, 0]
    R = (rs * rs.conjugate() + rp * rp.conjugate()) / 2

    # T should be R - 1
    ts = 1 / W_back_s[0, 0]
    tp = 1 / W_back_p[0, 0]
    T = cos_sub / cos_inc * n_sub * (
        ts * ts.conjugate() + tp * tp.conjugate()) / 2

    '''
    BACKWARD PROPAGATION
    '''
    partial_Ws_R = cuda.local.array((2, 2), dtype="complex128")
    partial_Wp_R = cuda.local.array((2, 2), dtype="complex128")
    partial_Ws_T = cuda.local.array((2, 2), dtype="complex128")
    partial_Wp_T = cuda.local.array((2, 2), dtype="complex128")

    # \partial_{W_{tot}} R = r^* \partial_{W_{tot}} r
    fill_arr(
        partial_Ws_R,
        rs.conjugate() * -(W_back_s[1, 0] / W_back_s[0, 0] ** 2),
        0,
        rs.conjugate() * 1 / W_back_s[0, 0],
        0
    )
    fill_arr(
        partial_Wp_R,
        rp.conjugate() * -(W_back_p[1, 0] / W_back_p[0, 0] ** 2),
        0,
        rp.conjugate() * 1 / W_back_p[0, 0],
        0
    )

    # \partial_{W_{tot}} T = t^* \partial_{W_{tot}} t
    fill_arr(
        partial_Ws_T,
        ts.conjugate() * (-1 / W_back_s[0, 0] ** 2) *
            (cos_sub / cos_inc * n_sub),
        0,
        0,
        0
    )
    fill_arr(
        partial_Wp_T,
        tp.conjugate() * (-1 / W_back_p[0, 0] ** 2) *
            (cos_sub / cos_inc * n_sub),
        0,
        0,
        0
    )

    W_front_s = cuda.local.array((2, 2), dtype="complex128")
    W_front_p = cuda.local.array((2, 2), dtype="complex128")
    Ms_inv = cuda.local.array((2, 2), dtype='complex128')
    Mp_inv = cuda.local.array((2, 2), dtype='complex128')
    partial_d_Ms = cuda.local.array((2, 2), dtype='complex128')
    partial_d_Mp = cuda.local.array((2, 2), dtype='complex128')
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
    calc_M_inv(Ms_inv, Mp_inv, cos_arr[0], n_arr[0], d[0], wl)
    mul_left(Ms_inv, W_back_s)  # M_0^-1 to left
    mul_left(Mp_inv, W_back_p)  # M_0^-1 to left

    for i in range(layer_number - 1):
        # M[i + 1] corresponds to i-th layer
        # (first layer with material A is the 0-th layer)

        calc_partial_d_M(partial_d_Ms, partial_d_Mp,
                         cos_arr[i % 2], n_arr[i % 2], d[i], wl)

        mul_to(W_front_s, partial_d_Ms, tmp_res_s)
        mul_to(tmp_res_s, W_back_s, tmp_res_s)

        mul_to(W_front_p, partial_d_Mp, tmp_res_p)
        mul_to(tmp_res_p, W_back_p, tmp_res_p)

        partial_d_Rs = hadm_mul(tmp_res_s, partial_Ws_R)
        partial_d_Rp = hadm_mul(tmp_res_p, partial_Wp_R)
        jacobi[thread_id, i] = \
            (partial_d_Rs * s_ratio + partial_d_Rp *
             p_ratio).real / (s_ratio + p_ratio)

        partial_d_Ts = hadm_mul(tmp_res_s, partial_Ws_T)
        partial_d_Tp = hadm_mul(tmp_res_p, partial_Wp_T)
        jacobi[thread_id + wls_size, i] = \
            (partial_d_Ts * s_ratio + partial_d_Tp *
             p_ratio).real / (s_ratio + p_ratio)

        # update W_back and W_front
        calc_M_inv(Ms_inv, Mp_inv, cos_arr[(
            i + 1) % 2], n_arr[(i + 1) % 2], d[i + 1], wl)
        mul_left(Ms_inv, W_back_s)  # M_0^-1 to left
        mul_left(Mp_inv, W_back_p)  # M_0^-1 to left

        calc_M(Ms, Mp, cos_arr[i % 2], n_arr[i % 2], d[i], wl)
        mul_right(W_front_s, Ms)  # M_0^-1 to left
        mul_right(W_front_p, Mp)  # M_0^-1 to left

    # special case: last layer!
    i = layer_number - 1
    calc_partial_d_M(partial_d_Ms, partial_d_Mp,
                     cos_arr[i % 2], n_arr[i % 2], d[i], wl)

    mul_to(W_front_s, partial_d_Ms, tmp_res_s)
    mul_to(tmp_res_s, W_back_s, tmp_res_s)

    mul_to(W_front_p, partial_d_Mp, tmp_res_p)
    mul_to(tmp_res_p, W_back_p, tmp_res_p)

    partial_d_Rs = hadm_mul(tmp_res_s, partial_Ws_R)
    partial_d_Rp = hadm_mul(tmp_res_p, partial_Wp_R)
    jacobi[thread_id, i] = \
        (partial_d_Rs * s_ratio + partial_d_Rp * p_ratio).real / (s_ratio + p_ratio)

    partial_d_Ts = hadm_mul(tmp_res_s, partial_Ws_T)
    partial_d_Tp = hadm_mul(tmp_res_p, partial_Wp_T)
    jacobi[thread_id + wls_size, i] = \
        (partial_d_Ts * s_ratio + partial_d_Tp * p_ratio).real / (s_ratio + p_ratio)


@cuda.jit
def calc_M(Ms, Mp, cosi, ni, di, wl):

    phi = 2 * cmath.pi * 1j * cosi * ni * di / wl
    coshi = cmath.cosh(phi)
    sinhi = cmath.sinh(phi)

    Ms[0, 0] = coshi
    Ms[0, 1] = sinhi / cosi / ni
    Ms[1, 0] = cosi * ni * sinhi
    Ms[1, 1] = coshi

    Mp[0, 0] = coshi
    Mp[0, 1] = sinhi * ni / cosi
    Mp[1, 0] = cosi / ni * sinhi
    Mp[1, 1] = coshi


@cuda.jit
def calc_M_inv(Ms, Mp, cosi, ni, di, wl):

    phi = 2 * cmath.pi * 1j * cosi * ni * di / wl
    coshi = cmath.cosh(phi)
    sinhi = cmath.sinh(phi)

    Ms[0, 0] = coshi
    Ms[0, 1] = -sinhi / cosi / ni
    Ms[1, 0] = -cosi * ni * sinhi
    Ms[1, 1] = coshi

    Mp[0, 0] = coshi
    Mp[0, 1] = -sinhi * ni / cosi
    Mp[1, 0] = -cosi / ni * sinhi
    Mp[1, 1] = coshi


@cuda.jit
def fill_arr(A, a00, a01, a10, a11):
    A[0, 0] = a00
    A[0, 1] = a01
    A[1, 0] = a10
    A[1, 1] = a11


@cuda.jit
def calc_partial_d_M(res_mat_s, res_mat_p, cosi, ni, di, wl):

    phi = 2 * cmath.pi * 1j * cosi * ni * di / wl
    coshi = cmath.cosh(phi)
    sinhi = cmath.sinh(phi)

    res_mat_s[0, 0] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl
    res_mat_s[0, 1] = 2 * cmath.pi * 1j * coshi / wl
    res_mat_s[1, 0] = 2 * cmath.pi * 1j * cosi ** 2 * ni ** 2 * coshi / wl
    res_mat_s[1, 1] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl

    res_mat_p[0, 0] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl
    res_mat_p[0, 1] = 2 * cmath.pi * 1j * ni ** 2 * coshi / wl
    res_mat_p[1, 0] = 2 * cmath.pi * 1j * cosi ** 2 * coshi / wl
    res_mat_p[1, 1] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl
