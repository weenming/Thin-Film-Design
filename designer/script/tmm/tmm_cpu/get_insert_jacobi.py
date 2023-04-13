from numpy import *
from tmm.tmm_cpu.get_n import get_n


def inserted_layers(d, materials, insert_layer_num, insert_position, insert_thickness=0.00001):
    # insert_layer_num 是插入前，被插入的层的index
    assert d[insert_layer_num] >= insert_position, 'insert position out of range of the inserted layer'
    if materials[insert_layer_num] == 'SiO2_OIC':
        insert_material = 'Nb2O5_OIC'
        inserted_material = 'SiO2_OIC'
    elif materials[insert_layer_num] == 'Nb2O5_OIC':
        insert_material = 'SiO2_OIC'
        inserted_material = 'Nb2O5_OIC'
    elif materials[insert_layer_num] == 'SiO2':
        insert_material = 'TiO2'
        inserted_material = 'SiO2'
    elif materials[insert_layer_num] == 'TiO2':
        insert_material = 'SiO2'
        inserted_material = 'TiO2'

    i = insert_layer_num
    materials_new = insert(materials, i, inserted_material)
    materials_new = insert(materials_new, i + 1, insert_material)
    d_new = insert(d, i, insert_position)
    d_new[i + 1] -= insert_position
    d_new = insert(d_new, i + 1, insert_thickness)
    return d_new, materials_new


def get_insert_jacobi_TFNN(wls, d, materials, insert_search_pts):
    return 0


def get_insert_jacobi_faster(wls, d, materials, insert_search_pts, insert_thickness=0.00001, theta0=7):
    '''
    :param wls: wavelengths
    :param d: thickness
    :param materials: materials
    :param insert_search_pts: calculate insert gradient at insert_search_pts points in every layer
    :param theta0 incident angle in degree
    :return: a Jacobi matrix with shape of 2*wls.shape[0] by insert_search_pts*layer_number
    '''
    # 薄膜数，不算基底
    layer_number = d.shape[0]
    # 入射角, 转化为degree
    theta0 = theta0 / 180 * pi

    # 遍历所有的待测波长，存到2N*1 spectrum里(先R再T)。
    jacobi = zeros((2 * wls.shape[0], insert_search_pts * layer_number))
    for wl_index in range(wls.shape[0]):
        # 折射率 返回layer_number+2个数字，因为基底和空气也在里面
        wl = wls[wl_index]
        n = get_n(wl, materials)
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

        Ms[layer_number + 1, 0:, 0:] = Dnplus1_s.copy()
        Mp[layer_number + 1, 0:, 0:] = Dnplus1_p.copy()
        Ms[0, 0:, 0:] = inv_D0_s.copy()
        Mp[0, 0:, 0:] = inv_D0_p.copy()
        for i in range(1, layer_number + 1):
            # M(k)是M_k,在dimension3上的长度是layer_number+2. 还要注意到M_n+1是最先乘的.
            # 只要是切片，就是ndarray，只有索引所有维度才能得到ndarray里的数字类型,注意到n[0]是空气,theta[0]也是空气，d[0]是第一层膜
            cosi = cosis[i, 0]
            phi = 2 * pi * 1j * cosi * n[i, 0] * d[i - 1] / wl

            ni = n[i, 0]
            coshi = cosh(phi)
            sinhi = sinh(phi)
            Ms[i, 0:, 0:] = array([[coshi, sinhi / cosi / ni], [cosi * ni * sinhi, coshi]])
            Mp[i, 0:, 0:] = array([[coshi, sinhi * ni / cosi], [cosi / ni * sinhi, coshi]])

        Ws_forward = zeros((layer_number + 2, 2, 2), dtype="complex_")
        Wp_forward = Ws_forward.copy()
        Ws_backward = Ws_forward.copy()
        Wp_backward = Ws_forward.copy()

        Ws_forward[0, 0:, 0:] = Ms[0, 0:, 0:].copy()
        Wp_forward[0, 0:, 0:] = Mp[0, 0:, 0:].copy()
        Ws_backward[layer_number+1, 0:, 0:] = Ms[layer_number+1, 0:, 0:]
        Wp_backward[layer_number+1, 0:, 0:] = Mp[layer_number+1, 0:, 0:]
        for i in range(1, layer_number + 2):
            Ws_forward[i, 0:, 0:] = dot(Ws_forward[i-1, 0:, 0:], Ms[i, 0:, 0:])
            Wp_forward[i, 0:, 0:] = dot(Wp_forward[i-1, 0:, 0:], Mp[i, 0:, 0:])
            Ws_backward[layer_number+1-i, 0:, 0:] = dot(Ms[layer_number+1-i, 0:, 0:],
                                                        Ws_backward[layer_number+2-i, 0:, 0:])
            Wp_backward[layer_number+1-i, 0:, 0:] = dot(Mp[layer_number+1-i, 0:, 0:],
                                                        Wp_backward[layer_number+2-i, 0:, 0:])
        rs = Ws_forward[layer_number+1, 1, 0] / Ws_forward[layer_number+1, 0, 0]
        rp = Wp_forward[layer_number+1, 1, 0] / Wp_forward[layer_number+1, 0, 0]
        ts = 1 / Ws_forward[layer_number+1, 0, 0]
        tp = 1 / Wp_forward[layer_number+1, 0, 0]
        R = ((rs * rs.conjugate() + rp * rp.conjugate()) / 2).real
        T = (cosis[layer_number + 1, 0] / cos(theta0) * n[layer_number + 1, 0] * (
                ts * ts.conjugate() + tp * tp.conjugate()) / 2).real

        for i in range(d.shape[0]):
            for j in range(insert_search_pts):
                d_new, materials_new = inserted_layers(d, materials, i, j*d[i]/insert_search_pts, insert_thickness)
                n = get_n(wl, materials_new)
                # 每一层的角度，theta[k]=theta_k,长度是layer_number+2
                cosis = zeros((layer_number + 4, 1), dtype='complex_')
                for k in range(layer_number + 4):
                    cosis[k] = sqrt(1 - (sin(theta0) / n[k]) ** 2)

                Ms_new = identity(2)
                Mp_new = identity(2)
                for l in range(i, i+3):
                    cosl = cosis[l+1, 0]
                    phi = 2 * pi * 1j * cosl * n[l+1, 0] * d_new[l] / wl
                    nl = n[l+1, 0]
                    coshl = cosh(phi)
                    sinhl = sinh(phi)
                    Ms_new = dot(Ms_new, array([[coshl, sinhl / cosl / nl], [cosl * nl * sinhl, coshl]]))
                    Mp_new = dot(Mp_new, array([[coshl, sinhl * nl / cosl], [cosl / nl * sinhl, coshl]]))

                Ws = dot(dot(Ws_forward[i, 0:, 0:], Ms_new), Ws_backward[i+2, 0:, 0:])
                Wp = dot(dot(Wp_forward[i, 0:, 0:], Mp_new), Wp_backward[i+2, 0:, 0:])

                rs = Ws[1, 0] / Ws[0, 0]
                rp = Wp[1, 0] / Wp[0, 0]
                ts = 1 / Ws[0, 0]
                tp = 1 / Wp[0, 0]
                R_new = (rs * rs.conjugate() + rp * rp.conjugate()) / 2
                T_new = cosis[-1, 0] / cos(theta0) * n[-1, 0] * (
                            ts * ts.conjugate() + tp * tp.conjugate()) / 2
                jacobi[wl_index, j + i * insert_search_pts] = (R_new.real - R)/insert_thickness
                jacobi[wl_index + wls.shape[0], j + i * insert_search_pts] = (T_new.real - T)/insert_thickness
    return jacobi


def get_insert_jacobi_faster_multi_inc(wls, d, materials, insert_search_pts, insert_thickness=0.00001, theta0=array([7])):
    jacobi = zeros((2 * theta0.shape[0] * wls.shape[0], insert_search_pts * d.shape[0]))
    for i in range(theta0.shape[0]):
        jacobi[i * 2 * wls.shape[0]: (i + 1) * 2 * wls.shape[0], :] = get_insert_jacobi_faster(wls, d, materials, insert_search_pts, insert_thickness, theta0[i])
    return jacobi

def get_insert_jacobi(wls, d, materials, insert_search_pts):
    from tmm.get_spectrum import get_spectrum
    """
    Calculates jacobi matrix (wls.shape[0] by insert_search_pts*layer_number)of insertion gradient. To be replaced by using TFNN
    :param wls:
    :param d:
    :param materials:
    :param insert_search_pts:
    :return:
    """
    layer_number = d.shape[0]
    insert_jacobi = zeros((2*wls.shape[0], layer_number * insert_search_pts))
    print('start searching for insertion place')
    for i in range(layer_number):
        for j in range(insert_search_pts):
            # # print(f'{(i*insert_search_pts+j)/(layer_number*insert_search_pts)} completed')
            # if materials[i] == 'SiO2_OIC':
            #     insert_material = 'Nb2O5_OIC'
            # else:
            #     insert_material = 'SiO2_OIC'
            #
            # materials_new = insert(materials, i, materials[i])
            # materials_new = insert(materials_new, i+1, insert_material)
            # d_new = insert(d, i, d[i]*j/insert_search_pts)
            # d_new[i+1] *= 1 - j/insert_search_pts
            # d_new = insert(d_new, i+1, 0.001)
            d_new, materials_new = inserted_layers(d, materials, i, d[i]*j/insert_search_pts)
            insert_jacobi[:, i*insert_search_pts+j] = (get_spectrum(wls, d_new, materials_new, theta0=60.) - get_spectrum(wls, d, materials, theta0=60.))[:, 0]*1e5

    return insert_jacobi

