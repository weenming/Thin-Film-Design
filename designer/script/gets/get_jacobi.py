import numpy as np
import cmath
from numba import cuda
from mat_lib import mul # multiply
from mat_lib import tps # transpose


def get_jacobi_simple(jacobi, wls, d, n_layers, n_sub, n_inc, inc_ang):
    """
    This function calculates the Jacobi matrix of a given TFNN. Back 
    propagation is implemented to acquire accurate result.

    Parameters:
        jacobi (2d np.array):
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
    spectrum = np.empty(wls_size)



def forward_and_back_propagation():
    # 遍历所有的待测波长 存到2N(R,T)*3M(d,n,k) jacobi matrix 里
    jacobi = zeros((2 * wls.shape[0], 3 * layer_number))
    for wl_index in range(wls.shape[0]):
        # 折射率 返回layer_number+2个数字，因为基底和空气也在里面
        wl = wls[wl_index]
        n = get_n(wl, materials)
        # 每一层的角度，theta[k]=theta_k,长度是layer_number+2
        cosis = zeros((layer_number + 2, 1), dtype='complex_')
        for i in range(layer_number + 2):
            cosis[i] = sqrt(1 - (sin(theta0) / n[i]) ** 2)

        # 正向传播
        # 基底和空气的传输矩阵
        inv_D0_s = array([[0.5, 0.5 / cos(theta0)], [0.5, -0.5 / cos(theta0)]])
        inv_D0_p = array([[0.5, 0.5 / cos(theta0)], [0.5, -0.5 / cos(theta0)]])
        Dnplus1_s = array([[1., 1.], [n[layer_number + 1, 0] * cosis[layer_number + 1, 0],
                                      -n[layer_number + 1, 0] * cosis[layer_number + 1, 0]]])
        Dnplus1_p = array(
            [[n[layer_number + 1, 0], n[layer_number + 1, 0]],
             [cosis[layer_number + 1, 0], -cosis[layer_number + 1, 0]]])
        #
        Ms = zeros((layer_number + 2, 2, 2), dtype="complex_")
        Mp = Ms.copy()
        Ws = zeros((layer_number + 2, 2, 2), dtype="complex_")
        Wp = Ws.copy()

        Ms[layer_number + 1, 0:, 0:] = Dnplus1_s
        Mp[layer_number + 1, 0:, 0:] = Dnplus1_p

        for i in range(1, layer_number + 1):
            # M(k)是M_k,在dimension3上的长度是layer_number+2. 还要注意到M_n+1是最先乘的.
            # 只要是切片，就是ndarray，只有索引所有维度才能得到ndarray里的数字类型,注意到n[0]是空气,theta[0]也是空气，d[0]是第一层膜
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

        Ws[0, 0:, 0:] = Ms[0, 0:, 0:].copy()
        Wp[0, 0:, 0:] = Mp[0, 0:, 0:].copy()
        for i in range(1, layer_number + 2):
            Ws[i, 0:, 0:] = dot(Ws[i - 1, 0:, 0:], Ms[i, 0:, 0:])
            Wp[i, 0:, 0:] = dot(Wp[i - 1, 0:, 0:], Mp[i, 0:, 0:])
        rs = Ws[layer_number + 1, 1, 0] / Ws[layer_number + 1, 0, 0]
        rp = Wp[layer_number + 1, 1, 0] / Wp[layer_number + 1, 0, 0]
        ts = 1 / Ws[layer_number + 1, 0, 0]
        tp = 1 / Wp[layer_number + 1, 0, 0]
        # 反向传播
        #
        partial_Ws_r = zeros((layer_number + 2, 2, 2), dtype="complex")
        partial_Ws_t = zeros((layer_number + 2, 2, 2), dtype="complex")
        partial_Wp_r = zeros((layer_number + 2, 2, 2), dtype="complex")
        partial_Wp_t = zeros((layer_number + 2, 2, 2), dtype="complex")
        partial_Ms_r = zeros((layer_number + 2, 2, 2), dtype="complex")
        partial_Ms_t = zeros((layer_number + 2, 2, 2), dtype="complex")
        partial_Mp_r = zeros((layer_number + 2, 2, 2), dtype="complex")
        partial_Mp_t = zeros((layer_number + 2, 2, 2), dtype="complex")
        # 对整个传输矩阵求导

        partial_Ws_r[layer_number + 1, 0:, 0:] = array(
            [[-Ws[layer_number + 1, 1, 0] / Ws[layer_number + 1, 0, 0] ** 2, 0], [1 / Ws[layer_number + 1, 0, 0], 0]])
        partial_Ws_t[layer_number + 1, 0:,
                     0:] = array([[-1 / Ws[layer_number + 1, 0, 0] ** 2, 0], [0, 0]])
        partial_Wp_r[layer_number + 1, 0:, 0:] = array(
            [[-Wp[layer_number + 1, 1, 0] / Wp[layer_number + 1, 0, 0] ** 2, 0], [1 / Wp[layer_number + 1, 0, 0], 0]])
        partial_Wp_t[layer_number + 1, 0:,
                     0:] = array([[-1 / Wp[layer_number + 1, 0, 0] ** 2, 0], [0, 0]])

        # 递推得到各个partialW
        for i in range(1, layer_number + 1):
            partial_Ws_r[layer_number + 1 - i, 0:, 0:] = dot(partial_Ws_r[layer_number + 2 - i, 0:, 0:],
                                                             Ms[layer_number + 2 - i, 0:, 0:].T)

            partial_Ms_r[layer_number + 1 - i, 0:, 0:] = dot(Ws[layer_number - i, 0:, 0:].T,
                                                             partial_Ws_r[layer_number + 1 - i, 0:, 0:])
            partial_Ws_t[layer_number + 1 - i, 0:, 0:] = dot(partial_Ws_t[layer_number + 2 - i, 0:, 0:],
                                                             Ms[layer_number + 2 - i, 0:, 0:].T)
            partial_Ms_t[layer_number + 1 - i, 0:, 0:] = dot(Ws[layer_number - i, 0:, 0:].T,
                                                             partial_Ws_t[layer_number + 1 - i, 0:, 0:])

            partial_Wp_r[layer_number + 1 - i, 0:, 0:] = dot(partial_Wp_r[layer_number + 2 - i, 0:, 0:],
                                                             Mp[layer_number + 2 - i, 0:, 0:].T)
            partial_Mp_r[layer_number + 1 - i, 0:, 0:] = dot(Wp[layer_number - i, 0:, 0:].T,
                                                             partial_Wp_r[layer_number + 1 - i, 0:, 0:])
            partial_Wp_t[layer_number + 1 - i, 0:, 0:] = dot(partial_Wp_t[layer_number + 2 - i, 0:, 0:],
                                                             Mp[layer_number + 2 - i, 0:, 0:].T)
            partial_Mp_t[layer_number + 1 - i, 0:, 0:] = dot(Wp[layer_number - i, 0:, 0:].T,
                                                             partial_Wp_t[layer_number + 1 - i, 0:, 0:])

            # 知道partialM[k] (k从1开始，到layer_number指各层膜。), 对应partial_x[k-1]

            cosi = cosis[layer_number + 1 - i, 0]
            phi = 2 * pi * 1j * cosi * \
                n[layer_number + 1 - i, 0] * d[layer_number - i] / wl
            di = d[layer_number - i]
            ni = n[layer_number + 1 - i, 0]
            coshi = cosh(phi)
            sinhi = sinh(phi)

            # partial_d_r
            jacobi[wl_index, layer_number - i] = (rp.conjugate() *
                                                  (partial_Mp_r[layer_number + 1 - i, 0:, 0:] *
                                                   array([[2 * pi * 1j * ni * cosi * sinhi,
                                                           2 * pi * 1j * ni ** 2 * coshi],
                                                          [2 * pi * 1j * cosi ** 2 * coshi,
                                                           2 * pi * 1j * ni * cosi * sinhi]])
                                                   ).sum() +
                                                  rs.conjugate() *
                                                  (partial_Ms_r[layer_number + 1 - i, 0:, 0:] *
                                                   array([[2 * pi * 1j * ni * cosi * sinhi, 2 * pi * 1j * coshi],
                                                          [2 * pi * 1j * cosi ** 2 * ni ** 2 * coshi,
                                                           2 * pi * 1j * ni * cosi * sinhi]])
                                                   ).sum()
                                                  ).real / wl
            # partial_d_t
            jacobi[wl_index + wls.shape[0], layer_number - i] = (cosis[layer_number + 1, 0] / cos(theta0) * n[layer_number + 1, 0]).real * \
                                                                (tp.conjugate() *
                                                                 (partial_Mp_t[layer_number + 1 - i, 0:, 0:] *
                                                                  array([[2 * pi * 1j * ni * cosi * sinhi,
                                                                          2 * pi * 1j * ni ** 2 * coshi],
                                                                         [2 * pi * 1j * cosi ** 2 * coshi,
                                                                          2 * pi * 1j * ni * cosi * sinhi]])
                                                                  ).sum() +
                                                                 ts.conjugate() *
                                                                 (partial_Ms_t[layer_number + 1 - i, 0:, 0:] *
                                                                  array([[2 * pi * 1j * ni * cosi * sinhi,
                                                                          2 * pi * 1j * coshi],
                                                                         [2 * pi * 1j * cosi ** 2 * ni ** 2 * coshi,
                                                                          2 * pi * 1j * ni * cosi * sinhi]])
                                                                  ).sum()
                                                                 ).real / wl
 return jacobi


def get_jacobi_multi_inc(wls, d, materials, theta0=array([7])):
    jacobi = zeros((2 * theta0.shape[0] * wls.shape[0], d.shape[0]))
    for i in range(theta0.shape[0]):
        jacobi[i * 2 * wls.shape[0]: (i + 1) * 2 * wls.shape[0], :] = get_jacobi(
            wls, d, materials, theta0[i])[:, 0:d.shape[0]]
    return jacobi
