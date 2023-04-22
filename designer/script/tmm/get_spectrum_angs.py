import numpy as np
import cmath
from numba import cuda
from tmm.mat_lib import mul_right, mul_left, tsp  # 2 * 2 matrix optr


def get_spectrum_simple(
    spectrum,
    wl,
    d,
    n_layers,
    n_sub,
    n_inc,
    inc_angs,
    s_ratio=1,
    p_ratio=1
):
    """
    This function calculates the reflectance and transmittance spectrum of a 
    non-polarized light (50% p-polarized and 50% s-polarized).

    It launches a CUDA kernel function after copying essential data to the GPU.

    Note that memory consumption of forward propagation does not scale with layer. 

    Arguments:
        spectrum (1d np.array):
            2 * wls.shape[0], type: float64
            pre-allocated memory space for returning spectrum 
        wls (1d np.array): 
            wls.shape[0]
            wavelengths of the target spectrum
        d (1d np.array):
            multi-layer thicknesses after last iteration
        n_layers (2d np.array): 
            wls.shape[0] \cross d.shape[0]. 
            refractive indices of each *layer*
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

    Returns:
        size: 2 \cross wls.shape[0] spectrum 
        (Reflectance spectrum + Transmittance spectrum).
    """
    # layer number of thin film, substrate not included
    layer_number = d.shape[0]
    # convert incident angle in degree to rad
    inc_angs_rad = inc_angs / 180 * np.pi
    # traverse all wl, save R and T to the 2N*1 np.array spectrum. [R, T]
    ang_size = inc_angs.shape[0]

    # TODO: move the copy of wls, n arr to outer loop
    # (caller of spec, for example LM optimizer)
    # Maybe allowing it to pass in additional device arr would be a good idea

    # copy wls, d, n_layers, n_sub, n_inc, inc_ang to GPU
    inc_angs_rad_device = cuda.to_device(inc_angs_rad)
    # d.astype('float64')
    d_device = cuda.to_device(d)
    # copy 2d arr into 1d as contiguous arr to save data transfer
    # BUG: this may be problematic. If the matrix it is copying from is row first
    # This copy would result in a new array full of "holes" which the kernel func
    # may not be aware of.
    n_A = n_layers[:, 0].copy()
    n_A_device = cuda.to_device(n_A)
    # may have only 1 layer.
    if layer_number == 1:
        # BUG: this might cause bug in 1 layer scenario. In the kernel function
        # nB and nA are used to construct a new array. The data types of these
        # arrays should thus be same
        # Spectrum objects the dtype of n is complex 128. keep same here

        # copying n_A seems to work but using np.empty(wls_size, dtype='complex128')
        # does not. I have literally no idea what is happening...
        n_B_device = cuda.to_device(n_A.copy())
    else:
        n_B = n_layers[:, 1].copy()
        n_B_device = cuda.to_device(n_B)
    n_sub_device = cuda.to_device(n_sub)
    n_inc_device = cuda.to_device(n_inc)
    # primitive transfer is not costly so I leave out inc_ang, wls_size and
    # layer_number

    # allocate space for R and T spec
    spectrum_device = cuda.device_array(ang_size * 2, dtype="float64")

    # invoke kernel
    block_size = 16  # threads per block
    grid_size = (ang_size + block_size - 1) // block_size  # blocks per grid

    forward_propagation_simple[grid_size, block_size](
        spectrum_device,
        wl,  # should not move wl to device, or 'typing error'
        d_device,
        n_A_device,
        n_B_device,
        n_sub_device,
        n_inc_device,
        inc_angs_rad_device,
        ang_size,
        layer_number,
        s_ratio,
        p_ratio
    )
    cuda.synchronize()
    # copy to pre-allocated space
    spectrum_device.copy_to_host(spectrum)


@cuda.jit
def forward_propagation_simple(
    spectrum,
    wl,
    d,
    n_A_arr,
    n_B_arr,
    n_sub_arr,
    n_inc_arr,
    inc_angs,
    ang_size,
    layer_number,
    s_ratio,
    p_ratio
):
    """
    Parameters:
        spectrum (cuda.device_array):
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
    if thread_id > ang_size - 1:
        return
    # each thread calculates one wl
    inc_ang = inc_angs[thread_id]

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

    # Allocate space for W. Fill with first term D_{0}^{-1}
    # TODO: add the influence of n of incident material (when not air)
    Ws = cuda.local.array((2, 2), dtype="complex128")
    Ws[0, 0] = 0.5
    Ws[0, 1] = 0.5 / cos_inc
    Ws[1, 0] = 0.5
    Ws[1, 1] = -0.5 / cos_inc

    Wp = cuda.local.array((2, 2), dtype="complex128")
    Wp[0, 0] = 0.5
    Wp[0, 1] = 0.5 / cos_inc
    Wp[1, 0] = 0.5
    Wp[1, 1] = -0.5 / cos_inc

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

    # retrieve R and T (calculate the factor before energy flux)
    # Note that spectrum is array on device
    rs = Ws[1, 0] / Ws[0, 0]
    rp = Wp[1, 0] / Wp[0, 0]
    R = (s_ratio * rs * rs.conjugate() + p_ratio * rp * rp.conjugate()) \
        / (s_ratio + p_ratio)
    spectrum[thread_id] = R.real

    # T should be R - 1
    ts = 1 / Ws[0, 0]
    tp = 1 / Wp[0, 0]
    T = cos_sub / cos_inc * n_sub * \
        (s_ratio * ts * ts.conjugate() + p_ratio * tp * tp.conjugate()) \
        / (s_ratio + p_ratio)
    spectrum[thread_id + ang_size] = T.real
