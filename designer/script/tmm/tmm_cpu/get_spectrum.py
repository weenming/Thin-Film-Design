from numpy import *
from tmm.tmm_cpu.get_n import get_n


def get_spectrum(wls, d, materials, n_indices=[], theta0=7., substrate=None):
    """
    This function calculates the reflectance and transmittance spectrum of a non-polarized light at 0 degrees

    :param wls: wavelengths of the target spectrum
    :param d: multi-layer thicknesses after last iteration
    :param n_indices: denotes the dispersion details of a layer=
    :param materials: material of a certain layer
    :return: [zN*1 by 1 matrix], N是采样的波长数目，z=2因为T和R都测了；M是层数，厚度，partial/partial n 的实部和虚部
    """
    # 薄膜数，不算基底
    layer_number = d.shape[0]
    theta0 = theta0 / 180 * pi
    # 遍历所有的待测波长，存到2N*1 spectrum里(先R再T)。
    spectrum = zeros((2 * wls.shape[0], 1))
    for wl_index in range(wls.shape[0]):
        # 折射率 返回layer_number+2个数字，因为基底和空气也在里面
        wl = wls[wl_index]
        n = get_n(wl, materials, substrate=substrate)
        # 每一层的角度，theta[k]=theta_k,长度是layer_number+2
        cosis = zeros((layer_number + 2, 1), dtype='complex_')
        for i in range(layer_number + 2):
            cosis[i] = sqrt(1 - (sin(theta0)/n[i])**2)

        # 正向传播
        inv_D0_s = array([[0.5, 0.5 / cos(theta0)], [0.5, -0.5 / cos(theta0)]])
        inv_D0_p = array([[0.5, 0.5 / cos(theta0)], [0.5, -0.5 / cos(theta0)]])
        Dnplus1_s = array([[1., 1.], [n[layer_number + 1, 0] * cosis[layer_number + 1, 0],
                                      -n[layer_number + 1, 0] * cosis[layer_number + 1, 0]]])
        Dnplus1_p = array(
            [[n[layer_number + 1, 0], n[layer_number + 1, 0]],
             [cosis[layer_number + 1, 0], -cosis[layer_number + 1, 0]]])
        Ms = zeros((layer_number + 2, 2, 2), dtype="complex_")
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
    return spectrum


def get_spectrum_multi_inc(wls, d, materials, theta0=array([7.]), substrate=None):
    spec = zeros((2 * theta0.shape[0] * wls.shape[0], 1))
    for i in range(theta0.shape[0]):
        spec[i * 2 * wls.shape[0]: (i + 1) * 2 * wls.shape[0], :] = get_spectrum(wls, d, materials, theta0=theta0[i], substrate=substrate)
    return spec