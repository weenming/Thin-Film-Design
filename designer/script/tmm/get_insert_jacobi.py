import numpy as np
import cmath
from numba import cuda
from tmm.mat_lib import mul_to # multiply
from tmm.mat_lib import tsp # transpose



def get_insert_jacobi_simple(
    jacobi, 
    wls, 
    d, 
    n_layers, 
    n_sub, 
    n_inc, 
    inc_ang, 
    total_layer_number=500
):
    """
    This function calculates the Jacobi matrix of a given TFNN. Back 
    propagation is implemented to acquire accurate result.
    NOTE: at maximum, MAX_LAYER_NUMBER layers supported
    NOTE: only difference from gd jacobi: more layers allowed for search.

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
        strides = (8 * layer_number, 8), 
        dtype="float64"
    )

    # invoke kernel
    block_size = 16 # threads per block
    grid_size = (wls_size + block_size - 1) // block_size # blocks per grid

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
        layer_number
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
    # Use local array which is private to thread. 
    # If shared array used, each block ~10KB, which is insufficient (?)
    cos_arr = cuda.local.array(2, dtype="complex128")
    cos_arr[0] = cos_A
    cos_arr[1] = cos_B

    n_arr = cuda.local.array(2, dtype="complex128")
    n_arr[0] = n_A
    n_arr[1] = n_B   

    '''
    FORWARD PROPAGATION
    '''

    # Allocate space for M 
    # NOTE: this implementation is shit. Need to find a way to make 
    # numba compile MAX_LAYER_NUMBER into constant...
    # Now I am MANUALLY EXPANDING THE CONSTANT INTO EVERY EXPRESSION
    # NOTE: Support MAX_LAYER_NUMBER layers for now.
    # also note that allocation of large space drastically affect the 
    # performance of the algorithm
    MAX_LAYER_NUMBER = 500
    # bad news: dynamic memory allocation not supported on GPU
    Ms = cuda.local.array((500 + 2, 2, 2), dtype="complex128") # NOTE: MAX_LAYER_NUMBER
    Mp = cuda.local.array((500 + 2, 2, 2), dtype="complex128") # NOTE: MAX_LAYER_NUMBER

    # Allocate space for W. Fill with first term D_{0}^{-1}
    # TODO: add the influence of n of incident material (when not air)
    Ws = cuda.local.array((500 + 2, 2, 2), dtype="complex128") # NOTE: MAX_LAYER_NUMBER
    Ws[0, 0, 0] = 0.5
    Ws[0, 0, 1] = 0.5 / cos_inc
    Ws[0, 1, 0] = 0.5
    Ws[0, 1, 1] = -0.5 / cos_inc
    
    Wp = cuda.local.array((500 + 2, 2, 2), dtype="complex128") # NOTE: MAX_LAYER_NUMBER
    Wp[0, 0, 0] = 0.5
    Wp[0, 0, 1] = 0.5 / cos_inc
    Wp[0, 1, 0] = 0.5
    Wp[0, 1, 1] = -0.5 / cos_inc

    for i in range(layer_number):
        cosi = cos_arr[i % 2]
        ni = n_arr[i % 2]
        phi = 2 * cmath.pi * 1j * cosi * ni * d[i] / wl

        coshi = cmath.cosh(phi)
        sinhi = cmath.sinh(phi)

        Ms[i + 1, 0, 0] = coshi
        Ms[i + 1, 0, 1] = sinhi / cosi / ni
        Ms[i + 1, 1, 0] = cosi * ni * sinhi
        Ms[i + 1, 1, 1] = coshi

        Mp[i + 1, 0, 0] = coshi
        Mp[i + 1, 0, 1] = sinhi * ni / cosi
        Mp[i + 1, 1, 0] = cosi / ni * sinhi
        Mp[i + 1, 1, 1] = coshi

        mul_to(Ws[i, :, :], Ms[i + 1, :, :], Ws[i + 1, :, :])
        mul_to(Wp[i, :, :], Mp[i + 1, :, :], Wp[i + 1, :, :])

    # construct the last term D_{n+1} 
    # technically this is merely D which is not M (D^{-2}PD)
    Ms[layer_number + 1, 0, 0] = 1.
    Ms[layer_number + 1, 0, 1] = 1.
    Ms[layer_number + 1, 1, 0] = n_sub * cos_sub
    Ms[layer_number + 1, 1, 1] = n_sub * cos_sub

    Mp[layer_number + 1, 0, 0] = n_sub
    Mp[layer_number + 1, 0, 1] = n_sub
    Mp[layer_number + 1, 1, 0] = cos_sub
    Mp[layer_number + 1, 1, 1] = cos_sub

    mul_to(Ws[layer_number, :, :], Ms[layer_number + 1, :, :], Ws[layer_number + 1, :, :])
    mul_to(Wp[layer_number, :, :], Mp[layer_number + 1, :, :], Wp[layer_number + 1, :, :])

    # retrieve R and T (calculate the factor before energy flux)
    # Note that spectrum is array on device
    rs = Ws[layer_number + 1, 1, 0] / Ws[layer_number + 1, 0, 0]
    rp = Wp[layer_number + 1, 1, 0] / Wp[layer_number + 1, 0, 0]
    R = (rs * rs.conjugate() + rp * rp.conjugate()) / 2

    # T should be R - 1
    ts = 1 / Ws[layer_number + 1, 0, 0]
    tp = 1 / Wp[layer_number + 1, 0, 0]
    T = cos_sub / cos_inc * n_sub * (
        ts * ts.conjugate() + tp * tp.conjugate()) / 2


    '''
    BACKWARD PROPAGATION
    '''
    # NOTE: integer 250 here should be constant MAX_LAYER_NUMBER
    partial_Ws_r = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    partial_Ws_t = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    partial_Wp_r = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    partial_Wp_t = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    partial_Ms_r = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    partial_Ms_t = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    partial_Mp_r = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    partial_Mp_t = cuda.local.array((500 + 2, 2, 2), dtype="complex128")
    
    # \partial_{W_{n + 1}} r or t (derivative from the last layer)
    partial_Ws_r[layer_number + 1, 0, 0] = -Ws[layer_number + 1, 1, 0] / Ws[layer_number + 1, 0, 0] ** 2
    partial_Ws_r[layer_number + 1, 0, 1] = 0
    partial_Ws_r[layer_number + 1, 1, 0] = 1 / Ws[layer_number + 1, 0, 0]
    partial_Ws_r[layer_number + 1, 1, 1] = 0
    
    partial_Ws_t[layer_number + 1, 0, 0] = -1 / Ws[layer_number + 1, 0, 0] ** 2
    partial_Ws_t[layer_number + 1, 0, 1] = 0
    partial_Ws_t[layer_number + 1, 1, 0] = 0
    partial_Ws_t[layer_number + 1, 1, 1] = 0
    
    partial_Wp_r[layer_number + 1, 0, 0] = -Wp[layer_number + 1, 1, 0] / Wp[layer_number + 1, 0, 0] ** 2
    partial_Wp_r[layer_number + 1, 0, 1] = 0
    partial_Wp_r[layer_number + 1, 1, 0] = 1 / Wp[layer_number + 1, 0, 0]
    partial_Wp_r[layer_number + 1, 1, 1] = 0
    
    partial_Wp_t[layer_number + 1, 0, 0] = -1 / Wp[layer_number + 1, 0, 0] ** 2
    partial_Wp_t[layer_number + 1, 0, 1] = 0
    partial_Wp_t[layer_number + 1, 1, 0] = 0
    partial_Wp_t[layer_number + 1, 1, 1] = 0

    # get \partial_W and thus \partial_M for every M
    tmp_matrix = cuda.local.array((2, 2), dtype="complex128") # for transpose
    for i in range(layer_number):
        # s-polarized
        tsp(Ms[layer_number + 1 - i, :, :], tmp_matrix)
        mul_to(
            partial_Ws_r[layer_number + 1 - i, :, :],
            tmp_matrix,
            partial_Ws_r[layer_number - i, :, :]
            )

        tsp(Ws[layer_number - i - 1, 0:, 0:], tmp_matrix)
        mul_to(
            tmp_matrix,
            partial_Ws_r[layer_number - i, 0:, 0:],
            partial_Ms_r[layer_number - i, 0:, 0:]
            )
        
        tsp(Ms[layer_number + 1 - i, 0:, 0:], tmp_matrix)
        mul_to(
            partial_Ws_t[layer_number + 1 - i, 0:, 0:],
            tmp_matrix,
            partial_Ws_t[layer_number - i, 0:, 0:]
            )

        tsp(Ws[layer_number - i - 1, 0:, 0:], tmp_matrix)
        mul_to(
            tmp_matrix,
            partial_Ws_t[layer_number - i, 0:, 0:],
            partial_Ms_t[layer_number - i, 0:, 0:]
            )

        # p-polarized
        tsp(Mp[layer_number + 1 - i, :, :], tmp_matrix)
        mul_to(
            partial_Wp_r[layer_number + 1 - i, :, :],
            tmp_matrix,
            partial_Wp_r[layer_number - i, :, :]
            )

        tsp(Wp[layer_number - i - 1, 0:, 0:], tmp_matrix)
        mul_to(
            tmp_matrix,
            partial_Wp_r[layer_number - i, 0:, 0:],
            partial_Mp_r[layer_number - i, 0:, 0:]
            )
        
        tsp(Mp[layer_number + 1 - i, 0:, 0:], tmp_matrix)
        mul_to(
            partial_Wp_t[layer_number + 1 - i, 0:, 0:],
            tmp_matrix,
            partial_Wp_t[layer_number - i, 0:, 0:]
            )

        tsp(Wp[layer_number - i - 1, 0:, 0:], tmp_matrix)
        mul_to(
            tmp_matrix,
            partial_Wp_t[layer_number - i, 0:, 0:],
            partial_Mp_t[layer_number - i, 0:, 0:]
            )

    # derive \partial_d r (t) from \partial_M r (t)

    for i in range(layer_number):
        # M[i + 1] corresponds to i-th layer 
        # (first layer with material A is the 0-th layer)
        cosi = cos_arr[i % 2]
        phi = 2 * cmath.pi * 1j * cosi * n_arr[i % 2] * d[i] / wl
        ni = n_arr[i % 2]
        coshi = cmath.cosh(phi)
        sinhi = cmath.sinh(phi)

        # partial_d R
        jacobi[thread_id, i] = \
            (
            rp.conjugate() * (
                partial_Mp_r[i + 1, 0, 0] * 2 * cmath.pi * 1j * ni * cosi * sinhi + 
                partial_Mp_r[i + 1, 0, 1] * 2 * cmath.pi * 1j * ni ** 2 * coshi +
                partial_Mp_r[i + 1, 1, 0] * 2 * cmath.pi * 1j * cosi ** 2 * coshi +
                partial_Mp_r[i + 1, 1, 1] * 2 * cmath.pi * 1j * ni * cosi * sinhi
            )
            + 
            rs.conjugate() * (
                partial_Ms_r[i + 1, 0, 0] * 2 * cmath.pi * 1j * ni * cosi * sinhi +
                partial_Ms_r[i + 1, 0, 1] * 2 * cmath.pi * 1j * coshi +
                partial_Ms_r[i + 1, 1, 0] * 2 * cmath.pi * 1j * cosi ** 2 * ni ** 2 * coshi +
                partial_Ms_r[i + 1, 1, 1] * 2 * cmath.pi * 1j * ni * cosi * sinhi
            )
            ).real / wl / 2 # div by 2 to normalize the sum of 2 polarizations
        
        # partial_d T
        jacobi[thread_id + wls_size, i] = \
            (cos_sub / cos_inc * n_sub).real * \
            (
            tp.conjugate() * (
                partial_Mp_t[i + 1, 0, 0] * 2 * cmath.pi * 1j * ni * cosi * sinhi + 
                partial_Mp_t[i + 1, 0, 1] * 2 * cmath.pi * 1j * ni ** 2 * coshi +
                partial_Mp_t[i + 1, 1, 0] * 2 * cmath.pi * 1j * cosi ** 2 * coshi +
                partial_Mp_t[i + 1, 1, 1] * 2 * cmath.pi * 1j * ni * cosi * sinhi
            )
            + 
            ts.conjugate() * (
                partial_Ms_t[i + 1, 0, 0] * 2 * cmath.pi * 1j * ni * cosi * sinhi +
                partial_Ms_t[i + 1, 0, 1] * 2 * cmath.pi * 1j * coshi +
                partial_Ms_t[i + 1, 1, 0] * 2 * cmath.pi * 1j * cosi ** 2 * ni ** 2 * coshi +
                partial_Ms_t[i + 1, 1, 1] * 2 * cmath.pi * 1j * ni * cosi * sinhi
            )
            ).real / wl / 2 # div by 2 to normalize the sum of 2 polarizations
