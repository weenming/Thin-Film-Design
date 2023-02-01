import numpy as np
from numba import cuda
from mat_lib import mul # multiply
from mat_lib import tps # transpose



def get_spectrum_simple(spectrum, wls, d, n_layers, n_sub, n_inc, inc_ang):
    """
    This function calculates the reflectance and transmittance spectrum of a non-polarized light at 0 degrees

    N是采样的波长数目, z=2因为T和R都测了; M是层数, 厚度, partial/partial n 的实部和虚部

    Arguments:
        spectrum (1d np.array):
            pre-allocated memory space for returning spectrum 
        wls (1d np.array): 
            wavelengths of the target spectrum
        d (1d np.array):
            multi-layer thicknesses after last iteration
        n_layers (2d np.array): 
            size: wls.shape[0] \cross d.shape[0]. refractive indices of each *layer*
        n_sub (1d np.array):
            refractive indices of the substrate
        n_inc (1d np.array):
            refractive indices of the incident material
        inc_ang (float):
            incident angle in degree

    Returns:
        size: 2 \cross wls.shape[0] spectrum (Reflectance spectrum + Transmittance spectrum).
    """
    # layer number of thin film, substrate not included
    layer_number = d.shape[0]
    # convert incident angle in degree to rad
    inc_ang = inc_ang / 180 * np.pi
    # traverse all wl, save R and T to the 2N*1 np.array spectrum. [R, T]
    wls_size = wls.shape[0]
    spectrum = np.empty(wls_size)

    # TODO: move the copy of wls, n arr to outer loop (caller of spec, for example 
    # LM optimizer) 
    # Maybe allowing it to pass in additional device arr would be a good idea

    # copy wls, d, n_layers, n_sub, n_inc, inc_ang to GPU
    wls_device = cuda.to_device(wls)
    d_device = cuda.to_device(d)
    n_layers_device = cuda.to_device(n_layers) # n = n_layers[tid, layer_index]
    n_sub_device = cuda.to_device(n_sub)
    n_inc_device = cuda.to_device(n_inc)
    # primitive transfer is not costly so I leave out inc_ang, wls_size and 
    # layer_number
    spectrum_device = cuda.device_array(wls_size) # only R spec
    
    # invoke kernel
    block_size = 32 # threads per block
    grid_size = (wls_size + block_size - 1) // block_size # blocks per grid
    forward_propagation[grid_size, block_size](
        spectrum_device,
        wls_device,
        d_device, 
        n_layers_device,
        n_sub_device,
        n_inc_device,
        inc_ang,
        wls_size,
        layer_number
    )
    # copy to pre-allocated space
    return spectrum_device.copy_to_host(spectrum)



@cuda.jit
def forward_propagation(spectrum, wls, d, n_layers, n_sub, n_inc, inc_ang, wl_size, layer_number):
    """
    n = n_layers[tid, layer_index]
    """
    thread_id = cuda.grid(1)
    # check this thread is valid
    if thread_id > wl_size:
        return
    # 折射率 返回layer_number+2个数字，因为基底和空气也在里面
    wl = wls[thread_id]
    
    # 每一层的角度，theta[k]=theta_k,长度是layer_number+2
    cosis = zeros((layer_number + 2, 1), dtype='complex128')
    for i in range(layer_number + 2):
        cosis[i] = sqrt(1 - (sin(theta0) / n[i])**2)

    # 正向传播
    inv_D0_s = array([[0.5, 0.5 / cos(theta0)], [0.5, -0.5 / cos(theta0)]])
    inv_D0_p = array([[0.5, 0.5 / cos(theta0)], [0.5, -0.5 / cos(theta0)]])
    Dnplus1_s = array([[1., 1.], [n[layer_number + 1, 0] * cosis[layer_number + 1, 0],
                                    -n[layer_number + 1, 0] * cosis[layer_number + 1, 0]]])
    Dnplus1_p = array(
        [[n[layer_number + 1, 0], n[layer_number + 1, 0]],
            [cosis[layer_number + 1, 0], -cosis[layer_number + 1, 0]]])
    Ms = zeros((layer_number + 2, 2, 2), dtype=np.complex128)
    Mp = Ms.copy()

    Ms[layer_number + 1, 0:, 0:] = Dnplus1_s
    Mp[layer_number + 1, 0:, 0:] = Dnplus1_p

    for i in range(1, layer_number + 1):
        # M(k)是M_k,在dimension3上的长度是layer_number+2. 还要注意到M_n+1是最先乘的.
        # 只要是切片，就是ndarray，只有索引所有维度才能得到ndarray里的数据类型,注意到n[0]是空气,theta[0]也是空气，d[0]是第一层膜
        cosi = cosis[i, 0]
        phi = 2 * pi * 1j * cosi * n[i, 0] * d[i - 1] / wl

        ni = n[i, 0]
        coshi = cosh(phi)
        sinhi = sinh(phi)
        # Ms[i, 0:, 0:] = array([[coshi, sinhi / cosi / ni], [cosi * ni * sinhi, coshi]])
        # Mp[i, 0:, 0:] = array([[coshi, sinhi * ni / cosi], [cosi / ni * sinhi, coshi]])
        Ms[i, 0, 0] = coshi
        Ms[i, 0, 1] = sinhi / cosi / ni
        Ms[i, 1, 0] = cosi * ni * sinhi
        Ms[i, 1, 1] = coshi

        Mp[i, 0, 0] = coshi
        Mp[i, 0, 1] = sinhi * ni / cosi
        Mp[i, 1, 0] = cosi / ni * sinhi
        Mp[i, 1, 1] = coshi

    Ms[0, 0:, 0:] = inv_D0_s.copy()
    Mp[0, 0:, 0:] = inv_D0_p.copy()

    Ws = Ms[0, 0:, 0:].copy()
    Wp = Mp[0, 0:, 0:].copy()
    for i in range(layer_number + 1):
        Ws = dot(Ws, Ms[i + 1, 0:, 0:])
        Wp = dot(Wp, Mp[i + 1, 0:, 0:])

    rs = Ws[1, 0] / Ws[0, 0]
    rp = Wp[1, 0] / Wp[0, 0]
    ts = 1 / Ws[0, 0]
    tp = 1 / Wp[0, 0]
    R = (rs * rs.conjugate() + rp * rp.conjugate()) / 2
    T = cosis[layer_number + 1, 0] / cos(theta0) * n[layer_number + 1, 0] * (
        ts * ts.conjugate() + tp * tp.conjugate()) / 2
    spectrum[wl_index, 0] = R.real
    spectrum[wl_index + wls.shape[0], 0] = T.real
