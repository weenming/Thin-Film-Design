import numpy as np
from gets.get_jacobi import get_jacobi
from gets.get_spectrum import get_spectrum
import time


def calculate_merit(d, materials, wls, target_spec):
    assert d.shape[0] == materials.shape[0], 'shape of d and materials don\'t align'
    f = get_spectrum(wls, d, materials, theta0=60.) - target_spec
    merit = np.sqrt((f ** 2 / wls.shape[0]).sum())
    return merit


def gradient_descent(wls, target_spec, d, materials, h_tol=1e-10):
    """
    Carries out the gradient descent process
    :param d:
    :return:
    """
    # 梯度下降
    print('start gradient descent')
    step_init = 1
    step = step_init
    layer_number = d.shape[0]

    J = get_jacobi(wls, d, materials, theta0=60.)[:, 0:layer_number]
    f = get_spectrum(wls, d, materials, theta0=60.) - target_spec
    g = np.dot(J.T, f)
    A = np.dot(J.T, J)
    nu = 2
    mu = 1
    merit = calculate_merit(d, materials, wls, target_spec)
    for gradient_descent_count in range(10000):
        start = time.time()
        # 验证梯度
        # if iter_count == 0:
        #     f_diff = get_spectrum(wls, d + 1e-3, materials)
        #     jacobi_diff = f_diff - get_spectrum(wls, d, materials)
        #     print(f'diff : {jacobi_diff*1e3}')
        #     print(f'derivative: {J}')

        # LM算法，下降
        J = get_jacobi(wls, d, materials, theta0=60.)[:, 0:layer_number]
        f = get_spectrum(wls, d, materials, theta0=60.) - target_spec
        h = np.dot(np.linalg.inv(A + mu * np.identity(layer_number)), -g)
        d_new = d + h[:, 0]
        # spectrum 是 2N by 1 向量

        F_d = (f ** 2).sum()
        # spectrum 是 6N by 1 向量
        spectrum_new = get_spectrum(wls, d_new, materials, theta0=60.)

        f_new = spectrum_new - target_spec
        F_dnew = (f_new ** 2).sum()
        rho = (F_d - F_dnew) / np.dot(h.T, mu * h - g).item()
        negative_count = 0
        if rho > 0:
            # 删掉负的
            for layer_test in range(layer_number):
                if d_new[layer_test] < 0:
                    negative_count += 1
                    d_deleted = d_new.copy()
                    materials_deleted = materials.copy()
                    if layer_test != layer_number - 1 and layer_test != 0:
                        d_deleted[layer_test - 1] += d_deleted[layer_test + 1]
                        d_deleted = np.delete(
                            d_deleted, [layer_test, layer_test + 1])
                        materials_deleted = np.delete(
                            materials_deleted, [layer_test, layer_test + 1])
                    else:
                        d_deleted = np.delete(d_deleted, layer_test)
                        materials_deleted = np.delete(
                            materials_deleted, layer_test)

            if negative_count == 1 and calculate_merit(d_deleted, materials_deleted, wls, target_spec) < merit:
                d = d_deleted.copy()
                materials = materials_deleted.copy()
                layer_number = d.shape[0]
                J = get_jacobi(wls, d, materials, theta0=60.)[
                    :, 0:layer_number]
                f = get_spectrum(wls, d, materials, theta0=60.) - target_spec
                g = np.dot(J.T, f)
                A = np.dot(J.T, J)
                nu = 2
                mu = 1

                insert_layer_num = -1
                # print(f'negative thickness! layer {layer_test} deleted')

            elif negative_count > 0:
                mu = mu * nu
                nu *= 2
                # print('descent rejected because negative or multiple layers deleted')

            elif negative_count == 0:
                d = d_new.copy()
                f = f_new
                g = np.dot(J.T, f)
                A = np.dot(J.T, J)
                mu = mu * \
                    np.amax(np.array([1 / 3, 1 - (2 * rho - 1) ** 3])).item()
                nu = 2
                merit = np.sqrt((f ** 2 / wls.shape[0]).sum())
                # print(f'accepted, merit function{merit}')
                # print(d)
        else:
            # print('descent rejected')
            mu = mu * nu
            nu = 2 * nu
        assert negative_count is not None, 'negative_count not defined'
        if np.amax(abs(h)).item() < h_tol and negative_count == 0:
            merit = np.sqrt(((get_spectrum(
                wls, d, materials, theta0=60.) - target_spec) ** 2 / wls.shape[0]).sum())
            # print(f'\noptimization of {layer_number}layer completed, merit function={merit}')
            break
        end = time.time()
        print(f'this iter takes {end - start} s')
    return d, materials, insert_layer_num, merit
