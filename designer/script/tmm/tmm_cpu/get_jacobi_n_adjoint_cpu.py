import numpy as np
import cmath
from tmm.tmm_cpu.mat_lib import mul_to, mul_right, mul_left, hadm_mul


def get_jacobi_free_form_cpu(
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
    NOTE: currently only real part of n is optimized

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

    for i in range(wls.shape[0]):
        forward_and_backward_propagation(
            i,
            jacobi,
            wls,
            d,
            n_layers,
            n_sub,
            n_inc,
            inc_ang_rad,
            wls_size,
            layer_number,
            s_ratio,
            p_ratio
        )


def forward_and_backward_propagation(
    thread_id,
    jacobi,
    wls,
    d,
    n_layers,
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

    # inc_ang is already in rad
    wl = wls[thread_id]
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

    W_back_s = np.zeros((2, 2), dtype="complex128")
    W_back_p = np.zeros((2, 2), dtype="complex128")

    fill_arr(W_back_s, 0.5, 0.5 / cos_inc / n_inc, 0.5, -0.5 / cos_inc / n_inc)
    fill_arr(W_back_p, 0.5 / n_inc, 0.5 / cos_inc, 0.5 / n_inc, -0.5 / cos_inc)

    Ms = np.zeros((2, 2), dtype="complex128")
    Mp = np.zeros((2, 2), dtype="complex128")

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
    rs = W_back_s[1, 0] / W_back_s[0, 0]
    rp = W_back_p[1, 0] / W_back_p[0, 0]

    # T should be R - 1
    ts = 1 / W_back_s[0, 0]
    tp = 1 / W_back_p[0, 0]

    '''
    BACKWARD PROPAGATION
    '''
    partial_Ws_R = np.zeros((2, 2), dtype="complex128")
    partial_Wp_R = np.zeros((2, 2), dtype="complex128")
    partial_Ws_T = np.zeros((2, 2), dtype="complex128")
    partial_Wp_T = np.zeros((2, 2), dtype="complex128")

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
            (cos_sub / cos_inc * n_sub).real,
        0,
        0,
        0
    )
    fill_arr(
        partial_Wp_T,
        tp.conjugate() * (-1 / W_back_p[0, 0] ** 2) *
            (cos_sub / cos_inc * n_sub).real,
        0,
        0,
        0
    )

    W_front_s = np.zeros((2, 2), dtype="complex128")
    W_front_p = np.zeros((2, 2), dtype="complex128")
    Ms_inv = np.zeros((2, 2), dtype='complex128')
    Mp_inv = np.zeros((2, 2), dtype='complex128')
    partial_n_Ms = np.zeros((2, 2), dtype='complex128')
    partial_n_Mp = np.zeros((2, 2), dtype='complex128')
    tmp_res_s = np.zeros((2, 2), dtype='complex128')
    tmp_res_p = np.zeros((2, 2), dtype='complex128')

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

    for i in range(layer_number - 1):
        # M[i + 1] corresponds to i-th layer
        # (first layer with material A is the 0-th layer)

        calc_partial_n_M(partial_n_Ms, partial_n_Mp,
                         n_inc, inc_ang, n_arr[i], d[i], wl)

        mul_to(W_front_s, partial_n_Ms, tmp_res_s)
        mul_to(tmp_res_s, W_back_s, tmp_res_s)

        mul_to(W_front_p, partial_n_Mp, tmp_res_p)
        mul_to(tmp_res_p, W_back_p, tmp_res_p)

        partial_n_Rs = hadm_mul(tmp_res_s, partial_Ws_R)
        partial_n_Rp = hadm_mul(tmp_res_p, partial_Wp_R)
        jacobi[thread_id, i] = \
            (partial_n_Rs * s_ratio + partial_n_Rp *
             p_ratio).real / (s_ratio + p_ratio)

        partial_n_Ts = hadm_mul(tmp_res_s, partial_Ws_T)
        partial_n_Tp = hadm_mul(tmp_res_p, partial_Wp_T)
        jacobi[thread_id + wls_size, i] = \
            (partial_n_Ts * s_ratio + partial_n_Tp *
             p_ratio).real / (s_ratio + p_ratio)

        # update W_back and W_front
        calc_M_inv(Ms_inv, Mp_inv, n_inc, inc_ang,
                   n_arr[i + 1], d[i + 1], wl)
        mul_left(Ms_inv, W_back_s)  # M_0^-1 to left
        mul_left(Mp_inv, W_back_p)  # M_0^-1 to left

        calc_M(Ms, Mp, n_inc, inc_ang, n_arr[i], d[i], wl)
        mul_right(W_front_s, Ms)  # M_0^-1 to left
        mul_right(W_front_p, Mp)  # M_0^-1 to left

    # special case: last layer!
    i = layer_number - 1
    calc_partial_n_M(partial_n_Ms, partial_n_Mp,
                     n_inc, inc_ang, n_arr[i], d[i], wl)

    mul_to(W_front_s, partial_n_Ms, tmp_res_s)
    mul_to(tmp_res_s, W_back_s, tmp_res_s)

    mul_to(W_front_p, partial_n_Mp, tmp_res_p)
    mul_to(tmp_res_p, W_back_p, tmp_res_p)

    partial_n_Rs = hadm_mul(tmp_res_s, partial_Ws_R)
    partial_n_Rp = hadm_mul(tmp_res_p, partial_Wp_R)
    jacobi[thread_id, i] = \
        (partial_n_Rs * s_ratio + partial_n_Rp * p_ratio).real / (s_ratio + p_ratio)

    partial_n_Ts = hadm_mul(tmp_res_s, partial_Ws_T)
    partial_n_Tp = hadm_mul(tmp_res_p, partial_Wp_T)
    jacobi[thread_id + wls_size, i] = \
        (partial_n_Ts * s_ratio + partial_n_Tp * p_ratio).real / (s_ratio + p_ratio)


def calc_M(Ms, Mp, n_inc, inc_ang, ni, di, wl):

    costheta = cmath.sqrt(
        1 - ((n_inc / ni) * cmath.sin(inc_ang)) ** 2)
    phi = 2 * cmath.pi * 1j * costheta * ni * di / wl
    coshi = cmath.cosh(phi)
    sinhi = cmath.sinh(phi)

    Ms[0, 0] = coshi
    Ms[0, 1] = sinhi / costheta / ni
    Ms[1, 0] = costheta * ni * sinhi
    Ms[1, 1] = coshi

    Mp[0, 0] = coshi
    Mp[0, 1] = sinhi * ni / costheta
    Mp[1, 0] = costheta / ni * sinhi
    Mp[1, 1] = coshi


def calc_M_inv(Ms, Mp, n_inc, inc_ang, ni, di, wl):
    costheta = cmath.sqrt(
        1 - ((n_inc / ni) * cmath.sin(inc_ang)) ** 2)

    phi = 2 * cmath.pi * 1j * costheta * ni * di / wl
    coshi = cmath.cosh(phi)
    sinhi = cmath.sinh(phi)

    Ms[0, 0] = coshi
    Ms[0, 1] = -sinhi / costheta / ni
    Ms[1, 0] = -costheta * ni * sinhi
    Ms[1, 1] = coshi

    Mp[0, 0] = coshi
    Mp[0, 1] = -sinhi * ni / costheta
    Mp[1, 0] = -costheta / ni * sinhi
    Mp[1, 1] = coshi


def fill_arr(A, a00, a01, a10, a11):
    A[0, 0] = a00
    A[0, 1] = a01
    A[1, 0] = a10
    A[1, 1] = a11


def calc_partial_n_M(res_mat_s, res_mat_p, n_inc, inc_ang, ni, di, wl):
    '''
        theta: incident angle at i-th layer
        phi: phase
    '''

    costheta = cmath.sqrt(
        1 - ((n_inc / ni) * cmath.sin(inc_ang)) ** 2)

    phi = 2 * cmath.pi * costheta * ni * di / wl
    cosphi = cmath.cos(phi)
    sinphi = cmath.sin(phi)
    pi = cmath.pi

    res_mat_s[0, 0] = - (2 * pi * di * sinphi) / (wl * costheta)
    res_mat_s[0, 1] = 1j * \
        ((2 * pi * di * cosphi) / (wl * ni * costheta ** 2) -
         sinphi / (ni ** 2 * costheta ** 3))
    res_mat_s[1, 0] = 1j * \
        ((2 * pi * di * cosphi * ni) / wl + sinphi / costheta)
    res_mat_s[1, 1] = - (2 * pi * di * sinphi) / (wl * costheta)

    res_mat_p[0, 0] = - (2 * pi * di * sinphi) / (wl * costheta)
    res_mat_p[0, 1] = 1j * (2 * pi * di * ni * costheta * cosphi +
                            (-1 + 2 * costheta**2) * wl * sinphi) / (costheta ** 3 * wl)
    res_mat_p[1, 0] = 1j * (2 * pi * di * ni * costheta * cosphi +
                            (1 - 2 * costheta**2) * wl * sinphi) / (costheta * wl * ni ** 2)
    res_mat_p[1, 1] = - (2 * pi * di * sinphi) / (wl * costheta)
