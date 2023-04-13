from numpy import *
from tmm.tmm_cpu.get_n import get_n


def get_jacobi(wls, d, materials, n_indices=[], theta0 = 7.):
    """
    This function calculates the gradient of the loss function against the thicknesses and refractive indices.

    The multilayer structure has M layers(not including substrate) and the spectrum is measured at N=wls.shape(0) different wavelengths,
    however, there are zN sampling points because both the transmittance and reflectance are measured

    :param wls: wavelengths of the target spectrum
    :param d: multi-layer thicknesses after last iteration
    :param n_parameters: refraction indices after last iteration
    :return: [zN by (3M)) matrix], N是采样的波长数目，z=2因为T和R都测了；M是层数，按照厚度，partial/partial n 的实部和虚部排列
    """
    # 薄膜数，不算基底
    layer_number = d.shape[0]
    # 入射角, 0°
    theta0 = theta0 / 180 * pi

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
        partial_Ws_t[layer_number + 1, 0:, 0:] = array([[-1 / Ws[layer_number + 1, 0, 0] ** 2, 0], [0, 0]])
        partial_Wp_r[layer_number + 1, 0:, 0:] = array(
            [[-Wp[layer_number + 1, 1, 0] / Wp[layer_number + 1, 0, 0] ** 2, 0], [1 / Wp[layer_number + 1, 0, 0], 0]])
        partial_Wp_t[layer_number + 1, 0:, 0:] = array([[-1 / Wp[layer_number + 1, 0, 0] ** 2, 0], [0, 0]])

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
            phi = 2 * pi * 1j * cosi * n[layer_number + 1 - i, 0] * d[layer_number - i] / wl
            di = d[layer_number - i]
            ni = n[layer_number + 1 - i, 0]
            coshi = cosh(phi)
            sinhi = sinh(phi)

            # partial_d R
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
            # partial_d T
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
            # # partial_n_r
            # jacobi[wl_index, layer_number - i + layer_number] = (rp.conjugate() *
            #                                                      (partial_Mp_r[layer_number + 1 - i, 0:, 0:] *
            #                                                       array([[2 * 1j * pi * di * sinh(phi) / cosi / wl, (
            #                                                               -2 * pi * 1j * di * ni ** 3 * cosh(
            #                                                           phi) / wl - (ni ** 2 - 2 * sin(
            #                                                           theta0) ** 2) * sinh(phi) / cosi) / (
            #                                                                       sin(theta0) ** 2 - ni ** 2)],
            #                                                              [(2 * pi * 1j * di * ni ** 3 * cosh(
            #                                                                  phi) / wl - (ni ** 2 - 2 * sin(
            #                                                                  theta0) ** 2) * sinh(phi) / cosi) / (
            #                                                                       ni ** 4),
            #                                                               2 * 1j * pi * di * sinh(
            #                                                                   phi) / cosi / wl]])).sum() +
            #                                                      rs.conjugate() *
            #                                                      (partial_Ms_r[layer_number + 1 - i, 0:, 0:] *
            #                                                       array([[2 * 1j * pi * di * sinh(phi) / cosi / wl,
            #                                                               (-1j * 2 * pi * di * ni ** 3 * cosh(
            #                                                                   phi) / wl + sinh(phi) / cosi) / (
            #                                                                       sin(theta0) ** 2 - ni ** 2)],
            #                                                              [1j * 2 * pi * di * ni ** 3 * cosh(
            #                                                                  phi) / wl + sinh(phi) / cosi,
            #                                                               2 * 1j * pi * di * sinh(
            #                                                                   phi) / cosi / wl]])).sum()
            #                                                      ).real
            # # partial_n_t
            # jacobi[wl_index + wls.shape[0], layer_number - i + layer_number] = (cosis[layer_number + 1, 0] / cos(theta0) * n[layer_number + 1, 0]).real *\
            #                                                                    (tp.conjugate() *
            #                                                                     (partial_Mp_t[layer_number + 1 - i, 0:,
            #                                                                      0:] *
            #                                                                      array([[2 * 1j * pi * di * sinh(
            #                                                                          phi) / cosi / wl,
            #                                                                              (
            #                                                                                      -2 * pi * 1j * di * ni ** 3 * cosh(
            #                                                                                  phi) / wl - (
            #                                                                                              ni ** 2 - 2 * sin(
            #                                                                                          theta0) ** 2) * sinh(
            #                                                                                  phi) / cosi) / (
            #                                                                                      sin(theta0) ** 2 - ni** 2)],
            #                                                                             [(
            #                                                                                      2 * pi * 1j * di * ni ** 3 * cosh(
            #                                                                                  phi) / wl - (
            #                                                                                              ni ** 2 - 2 * sin(
            #                                                                                          theta0) ** 2) * sinh(
            #                                                                                  phi) / cosi) / (
            #                                                                                      ni ** 4),
            #                                                                              2 * 1j * pi * di * sinh(
            #                                                                                  phi) / cosi / wl]])).sum() +
            #                                                                     ts.conjugate() *
            #                                                                     (partial_Ms_t[layer_number + 1 - i, 0:,
            #                                                                      0:] *
            #                                                                      array([[2 * 1j * pi * di * sinh(
            #                                                                          phi) / cosi / wl,
            #                                                                              (
            #                                                                                      -1j * 2 * pi * di * ni ** 3 * cosh(
            #                                                                                  phi) / wl + sinh(
            #                                                                                  phi) / cosi) / (
            #                                                                                      sin(theta0) ** 2 - ni ** 2)],
            #                                                                             [
            #                                                                                 1j * 2 * pi * di * ni ** 3 * cosh(
            #                                                                                     phi) / wl + sinh(
            #                                                                                     phi) / cosi,
            #                                                                                 2 * 1j * pi * di * sinh(
            #                                                                                     phi) / cosi / wl]])).sum()
            #                                                                     ).real
            # # partial_k_r
            # jacobi[wl_index, layer_number - i + 2 * layer_number] = (rp.conjugate() *
            #                                                          (partial_Mp_r[layer_number + 1 - i, 0:, 0:] *
            #                                                           array([[2 * pi * di * sinh(phi) / cosi / wl,
            #                                                                   (-2 * pi * di * ni ** 3 * cosh(
            #                                                                       phi) / wl + 1j * (ni ** 2 - 2 * sin(
            #                                                                       theta0) ** 2) * sinh(phi) / cosi) / (
            #                                                                           sin(theta0) ** 2 - ni ** 2)],
            #                                                                  [(2 * pi * di * ni ** 3 * cosh(
            #                                                                      phi) / wl + 1j * (ni ** 2 - 2 * sin(
            #                                                                      theta0) ** 2) * sinh(phi) / cosi) / (
            #                                                                           ni ** 4),
            #                                                                   2 * pi * di * sinh(
            #                                                                       phi) / cosi / wl]])).sum() +
            #                                                          rs.conjugate() *
            #                                                          (partial_Ms_r[layer_number + 1 - i, 0:, 0:] *
            #                                                           array([[2 * pi * di * sinh(phi) / cosi / wl,
            #                                                                   (-2 * pi * di * ni ** 3 * cosh(
            #                                                                       phi) / wl - 1j * sinh(phi) / cosi) / (
            #                                                                           sin(theta0) ** 2 - ni ** 2)],
            #                                                                  [2 * pi * di * ni ** 3 * cosh(
            #                                                                      phi) / wl - 1j * sinh(phi) / cosi,
            #                                                                   2 * pi * di * sinh(
            #                                                                       phi) / cosi / wl]])).sum()
            #                                                          ).real
            # # partial_k_t
            # jacobi[wl_index + wls.shape[0], layer_number - i + 2 * layer_number] = (cosis[layer_number + 1, 0] / cos(theta0) * n[layer_number + 1, 0]).real *\
            #                                                                        (tp.conjugate() *
            #                                                                         (partial_Mp_t[layer_number + 1 - i,
            #                                                                          0:, 0:] *
            #                                                                          array([[2 * pi * di * sinh(
            #                                                                              phi) / cosi / wl,
            #                                                                                  (
            #                                                                                          -2 * pi * di * ni ** 3 * cosh(
            #                                                                                      phi) / wl + 1j * (
            #                                                                                                  ni ** 2 - 2 * sin(
            #                                                                                              theta0) ** 2) * sinh(
            #                                                                                      phi) / cosi) / (
            #                                                                                          sin(theta0) ** 2 - ni ** 2)],
            #                                                                                 [(
            #                                                                                          2 * pi * di * ni ** 3 * cosh(
            #                                                                                      phi) / wl + 1j * (
            #                                                                                                  ni ** 2 - 2 * sin(
            #                                                                                              theta0) ** 2) * sinh(
            #                                                                                      phi) / cosi) / (
            #                                                                                          ni ** 4),
            #                                                                                  2 * pi * di * sinh(
            #                                                                                      phi) / cosi / wl]])).sum() +
            #                                                                         ts.conjugate() *
            #                                                                         (partial_Ms_t[layer_number + 1 - i,
            #                                                                          0:, 0:] *
            #                                                                          array([[2 * pi * di * sinh(
            #                                                                              phi) / cosi / wl,
            #                                                                                  (
            #                                                                                          -2 * pi * di * ni ** 3 * cosh(
            #                                                                                      phi) / wl - 1j * sinh(
            #                                                                                      phi) / cosi) / (
            #                                                                                          sin(theta0) ** 2 - ni ** 2)],
            #                                                                                 [
            #                                                                                     2 * pi * di * ni ** 3 * cosh(
            #                                                                                         phi) / wl - 1j * sinh(
            #                                                                                         phi) / cosi,
            #                                                                                     2 * pi * di * sinh(
            #                                                                                         phi) / cosi / wl]])).sum()
            #                                                                         ).real

    return jacobi

def get_jacobi_multi_inc(wls, d, materials, theta0=array([7])):
    jacobi = zeros((2 * theta0.shape[0] * wls.shape[0], d.shape[0]))
    for i in range(theta0.shape[0]):
        jacobi[i * 2 * wls.shape[0]: (i + 1) * 2 * wls.shape[0], :] = get_jacobi(wls, d, materials, theta0[i])[:, 0:d.shape[0]]
    return jacobi
