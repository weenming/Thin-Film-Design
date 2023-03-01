import numpy as np
from numba import cuda
from film import FilmSimple
from gets.get_spectrum import forward_propagation_simple 

def get_W_everywhere(film: FilmSimple):
    '''
    This function calculates the electric field distibution (at interfaces
    given by transfer matrices) of a non-polarized light (50% p-polarized 
    and 50% s-polarized).
    '''

    for spec in film.get_all_spec_list(): # stack specs
        pass
    return

def get_W_before_ith_layer(film:FilmSimple, i):
    '''
    This function gets W_i which is defined as in gets.get_spectrum, the product
    of transfer matrices of layers before i.
    $W_i^{before} \def D_{inc}^{-1} \prod_{j=1}^{i-1} D_j P_j D_j^{-1}$

    Implementation: to save time of copying a large array from GPU to memory, 
    we choose to perform 2 forward propagations for each layer to substitute.
    I am not sure if this implementation would be faster...Better than search for 
    sure, though.
    '''

def get_W_after_ith_layer(film:FilmSimple, i):
    '''
    This function gets W_i which is defined *DIFFERENT* from gets.get_spectrum, 
    the product of transfer matrices of layers before i.
    $W_i^{after} \def (\prod_{j=i+1}^{n} D_j P_j D_j^{-1}) D_{sub}$

    Implementation: same as above
    '''

def _launch_propagation(spectrum, wls, d, n_layers, n_sub, n_inc, inc_ang):
    """
    Returns:
        size: 2 \cross wls.shape[0] spectrum 
        (Reflectance spectrum + Transmittance spectrum).
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

    # allocate space for R and T spec
    spectrum_device = cuda.device_array(wls_size * 2, dtype="float64")

    # invoke kernel
    block_size = 16 # threads per block
    grid_size = (wls_size + block_size - 1) // block_size # blocks per grid
    
    forward_propagation_simple[grid_size, block_size](
        spectrum_device,
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
    spectrum_device.copy_to_host(spectrum)
    


def get_E_at_i(spectrum, wls, d, n_layers, n_sub, n_inc, inc_ang):
    return