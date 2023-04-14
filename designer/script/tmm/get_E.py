import numpy as np
from numba import cuda
from film import TwoMaterialFilm
import cmath
from tmm.mat_lib import mul_right, tsp  # 2 * 2 matrix optr


def get_E(wls, d, n_layers, n_sub, n_inc, inc_ang):
    """

    Parameters:
        i_start: index to start calculating Wi. If -1, incidence material
        i_end: index to end. If n, substrate
            consisitent with the index of the d array
    Returns:
        2 \corss 2 \cross sum of len(wls) for this specturm  (W_i for every wl)
    """
    E_spec = np.empty((wls.shape[0] * 2, 2), dtype="complex128")
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
    n_A = n_layers[:, 0].copy(order="C")
    n_A_device = cuda.to_device(n_A)
    # may have only 1 layer.
    if layer_number == 1:
        n_B_device = cuda.to_device(np.empty(wls_size))
    else:
        n_B = n_layers[:, 1].copy(order="C")
        n_B_device = cuda.to_device(n_B)
    n_sub_device = cuda.to_device(n_sub)
    n_inc_device = cuda.to_device(n_inc)
    # primitive transfer is not costly so I leave out inc_ang, wls_size and
    # layer_number

    # allocate space for E spec on GPU
    E_spec_device = cuda.device_array((wls_size * 2, 2), dtype="complex128")

    # invoke kernel
    block_size = 16  # threads per block
    grid_size = (wls_size + block_size - 1) // block_size  # blocks per grid

    forward_propagation_simple_E[grid_size, block_size](
        E_spec_device,
        wls_device,
        d_device,
        n_A_device,
        n_B_device,
        n_sub_device,
        n_inc_device,
        inc_ang_rad,
        wls_size,
        layer_number
    )
    cuda.synchronize()
    # copy to pre-allocated space
    E_spec_device.copy_to_host(E_spec)
    return E_spec


@cuda.jit
def forward_propagation_simple_E(E_spec, wls, d, n_A_arr, n_B_arr,
                                 n_sub_arr, n_inc_arr, inc_ang, wls_size, layer_number):
    """
    Parameters:
        E_spec (cuda.device_array):
            device array for storing data
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
    # incident angle in each layer. Snell's law: n_a sin(phi_a) = n_b sin(phi_b)
    cos_A = cmath.sqrt(1 - ((n_inc / n_A) * cmath.sin(inc_ang)) ** 2)
    cos_B = cmath.sqrt(1 - ((n_inc / n_B) * cmath.sin(inc_ang)) ** 2)
    cos_inc = cmath.cos(inc_ang)
    cos_sub = cmath.sqrt(1 - ((n_inc / n_sub) * cmath.sin(inc_ang)) ** 2)

    # choose cos from arr of size 2. Use local array which is private to thread
    cos_arr = cuda.local.array(2, dtype="complex128")
    cos_arr[0] = cos_A
    cos_arr[1] = cos_B

    n_arr = cuda.local.array(2, dtype="complex128")
    n_arr[0] = n_A
    n_arr[1] = n_B

    # Allocate space for M
    Ms = cuda.local.array((2, 2), dtype="complex128")
    Mp = cuda.local.array((2, 2), dtype="complex128")

    # Allocate space for W.
    Ws = cuda.local.array((2, 2), dtype="complex128")
    Wp = cuda.local.array((2, 2), dtype="complex128")

    # Initialize W according to i_start
    # TODO: add the influence of n of incident material (when not air)

    Ws[0, 0] = 0.5
    Ws[0, 1] = 0.5 / cos_inc
    Ws[1, 0] = 0.5
    Ws[1, 1] = -0.5 / cos_inc

    Wp[0, 0] = 0.5
    Wp[0, 1] = 0.5 / cos_inc
    Wp[1, 0] = 0.5
    Wp[1, 1] = -0.5 / cos_inc

    # forward propagation
    for i in range(layer_number):
        cosi = cos_arr[i % 2]
        ni = n_arr[i % 2]
        phi = 2 * cmath.pi * 1j * cosi * ni * d[i] / wl

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

        mul_right(Ws, Ms)
        mul_right(Wp, Mp)

    # construct the last term D_{n+1}
    # technically this is merely D which is not M (D^{-2}PD)

    Ms[0, 0] = 1.
    Ms[0, 1] = 1.
    Ms[1, 0] = n_sub * cos_sub
    Ms[1, 1] = n_sub * cos_sub

    Mp[0, 0] = n_sub
    Mp[0, 1] = n_sub
    Mp[1, 0] = cos_sub
    Mp[1, 1] = cos_sub

    mul_right(Ws, Ms)
    mul_right(Wp, Mp)

    for i in [0, 1]:
        E_spec[thread_id, i] = Ws[i, 0]  # s-polarized
        E_spec[thread_id + wls_size, i] = Wp[i, 0]  # p-polarized
