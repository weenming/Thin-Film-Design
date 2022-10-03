from numpy import *


def get_n(wl, materials, n_indices=[]):
    """
    从色散参数算折射率, wl in nm
    :param wl:
    :param n_indices:
    :return: M+1 *1 array, complex refractive indices (including the substrate)
    """
    # n的最后一行放的是基底Sellmeier色散的6个参数B1,B2...C2,C3，前面放Cauchy模型的4个参数A0,A1,A2,Delta m和吸收系数D1,D2, D3

    # 换算单位，从纳米到微米
    wl = wl*1e-3
    layer_number = materials.shape[0]
    # 吸收的系数，1是弱，2是正常，3强
    k_parameters = {1: [4e5, 56.0, 1e-10], 2: [3e5, 50.0, 1e-8], 3: [4e4, 40.0, 1e-6]}
    # 为了存复数，创建ndarray的时候要指定
    n = zeros((layer_number+2, 1), dtype="complex_")
    # 从色散参数算复数折射率
    n_SiO2 = sqrt(1.28604141 + 1.07044083 * wl**2 / (wl**2 - 1.00585997*1e4) + 1.10202242 * wl**2 / (wl**2 - 1e8))
    n_TiO2 = sqrt(5.913+0.2441/(wl**2 * 1e-6 - 0.0803))# sb 了，之前重复算了好多

    k_parameters = {1: [4e5, 56.0, 1e-10], 2: [3e5, 50.0, 1e-8], 3: [4e4, 40.0, 1e-6]}
    [D1, D2, D3] = k_parameters[2]
    delta_m = 0
    A0, A1, A2 = 2.218485, 0.021827, 3.99968e-3
    n_Nb2O5_OIC = A0 + A1 / wl ** 2 + A2 / wl ** 4 - (D1 * exp(-D2 * wl) + D3) * 1j + delta_m

    A0, A1, A2 = 1.460472, 0.0, 4.9867e-4
    n_SiO2_OIC = A0 + A1 / wl ** 2 + A2 / wl ** 4 + delta_m

    for i in range(layer_number + 2):
        if i == 0:
            # 入射介质：空气
            n[i] = 1
        elif i == layer_number+1:
            # 基底: 空气, OIC
            B1, B2, B3, C1, C2, C3 = 1.03961, 0.23179, 1.01047, 0.006, 0.02, 103.56
            n[i] = sqrt(
                1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2) + B3 * wl ** 2 / (wl ** 2 - C3))
            # n[i] = 1
        else:
            if materials[i - 1] == 'Nb2O5_OIC':
                n[i] = n_Nb2O5_OIC
            elif materials[i-1] == 'SiO2_OIC':
                n[i] = n_SiO2_OIC
            elif materials[i-1] == 'TiO2':
                # 430~1530nm
                n[i] = n_TiO2
            elif materials[i-1] == 'SiO2':
                # 198~2050nm
                n[i] = n_SiO2
    return n
