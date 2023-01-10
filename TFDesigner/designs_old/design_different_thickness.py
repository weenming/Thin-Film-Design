import sys

sys.path.append('..')

import numpy as np
from gets.get_jacobi import get_jacobi
from gets.get_spectrum import get_spectrum
from gets.get_n import get_n
import gets.get_insert_jacobi as get_insert_jacobi
import matplotlib.pyplot as plt
import pandas as pd
import time

# 用黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False


def plot_layer_thickness(d, materials, insert_layer_num=-1, last_insert_layer_thickness=60):
    d_plot = 0
    layer_number = d.shape[0]
    for i in range(layer_number):
        # 以600nm的折射率表示
        n_plot = get_n(600, materials)[i + 1].real
        n_last_plot = get_n(600, materials)[i].real
        if i == insert_layer_num:
            plt.plot([d_plot, d_plot + d[i]], [n_plot, n_plot], color='red',
                     label=f'thickness={last_insert_layer_thickness}')
        else:
            plt.plot([d_plot, d_plot + d[i]], [n_plot, n_plot], color='blue')
        if i != 0:
            plt.plot([d_plot, d_plot], [n_plot, n_last_plot], color='blue')
        d_plot += d[i]
    plt.legend()
    plt.title(f'layer thickness, {layer_number} layers')
    plt.show()
    return


def plot_spectrum(wls, target_spec, d, materials, layer_number, merit, insert_count):
    """
    Plot the reflection spectrum of the given layer structure and wls

    :param wls: 1-d numpy array of the evaluated wl points
    :param target_spec: 1-d numpy array, R at wl points specified by wls
        for comparison, plot also the target spectrum
    :param d: 1-d numpy array, thicknesses of the multilayer structure
    """
    assert wls.shape == target_spec.shape, 'target_spec shape do not align'

    print(f'insert_count={insert_count}, d={d}')
    fig_spec, (axR, axT) = plt.subplots(1, 2, sharey=True)
    axR.plot(wls, get_spectrum(wls, d, materials, theta0=60.)[0:wls.shape[0]], label='fit')
    axR.plot(wls, target_spec[0:wls.shape[0]], label='target')
    axR.legend()
    axR.set_xlabel('wl/nm')
    axR.set_title(f'R')

    axT = plt.subplot(122)
    axT.plot(wls, get_spectrum(wls, d, materials, theta0=60.)[wls.shape[0]:], label='fit')
    axT.plot(wls, target_spec[wls.shape[0]:], label='target')
    axT.legend()
    axT.set_xlabel('wl/nm')
    axT.set_title('T')

    fig_spec.suptitle(f'{layer_number} layers, merit func = {merit}')
    fig_spec.show()
    return


def descent_process(layer_number, merit, merits=np.array([]), layer_numbers=np.array([]), optical_thickness_ratio=None,
                    do_plot=False):
    """

    :param layer_number:
    :param merit:
    :param merits:
    :param layer_numbers:
    :param optical_thickness_ratio:
    :param do_plot:
    :return:
    """
    merits = np.append(merits, merit)
    layer_numbers = np.append(layer_numbers, layer_number)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(merits.shape[0]), np.log(merits), label='merits', color='orange')
    ax2.plot(range(merits.shape[0]), layer_numbers, label='layer number', color='steelblue')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    fig.suptitle('5layer_60degree')
    plt.savefig(f'../results/training_plot_different_init_thickness/merit_to_layer_thickness_ratio_{optical_thickness_ratio}.png')
    if do_plot:
        plt.show()

    def next_descent_process(layer_number, merit):
        return descent_process(layer_number, merit, merits, layer_numbers, optical_thickness_ratio)

    return next_descent_process


def gradient_descent(wls, target_spec, d, materials, insert_layer_num):
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
                        d_deleted = np.delete(d_deleted, [layer_test, layer_test + 1])
                        materials_deleted = np.delete(materials_deleted, [layer_test, layer_test + 1])
                    else:
                        d_deleted = np.delete(d_deleted, layer_test)
                        materials_deleted = np.delete(materials_deleted, layer_test)

            if negative_count == 1 and calculate_merit(d_deleted, materials_deleted, wls, target_spec) < merit:
                d = d_deleted.copy()
                materials = materials_deleted.copy()
                layer_number = d.shape[0]
                J = get_jacobi(wls, d, materials, theta0=60.)[:, 0:layer_number]
                f = get_spectrum(wls, d, materials, theta0=60.) - target_spec
                g = np.dot(J.T, f)
                A = np.dot(J.T, J)
                nu = 2
                mu = 1

                insert_layer_num = -1
                print(f'negative thickness! layer {layer_test} deleted')

            elif negative_count > 0:
                mu = mu * nu
                nu *= 2
                print('descent rejected because negative or multiple layers deleted')

            elif negative_count == 0:
                d = d_new.copy()
                f = f_new
                g = np.dot(J.T, f)
                A = np.dot(J.T, J)
                mu_new = mu * np.amax(np.array([1 / 3, 1 - (2 * rho - 1) ** 3])).item()
                if mu_new:  # avoid underflow of mu
                    mu = mu_new
                nu = 2
                merit = np.sqrt((f ** 2 / wls.shape[0]).sum())
                print(f'accepted, merit function{merit}')
                # print(d)
        else:
            print('descent rejected')
            mu = mu * nu
            nu = 2 * nu
        assert negative_count is not None, 'negative_count not defined'
        if np.amax(abs(h)).item() < 1e-10 and negative_count == 0:
            merit = np.sqrt(((get_spectrum(wls, d, materials, theta0=60.) - target_spec) ** 2 / wls.shape[0]).sum())
            print(f'\noptimization of {layer_number}layer completed, merit function={merit}')
            break
    return d, materials, insert_layer_num, merit


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
    materials_new = np.insert(materials, i, inserted_material)
    materials_new = np.insert(materials_new, i + 1, insert_material)
    d_new = np.insert(d, i, insert_position)
    d_new[i + 1] -= insert_position
    d_new = np.insert(d_new, i + 1, insert_thickness)
    return d_new, materials_new


def insert_1_layer(wls, target_spec, d, materials, insert_search_pts):
    """
    find the layer and position to insert the new layer
    :param d:
    :param materials:
    :param insert_search_pts:
    :return: inserted d, materials and the index of the inserted layer
    """
    print('start searching for insertion position')
    insert_gradient = np.dot(
        get_insert_jacobi.get_insert_jacobi_faster(wls, d, materials, insert_search_pts, theta0=60.).T,
        get_spectrum(wls, d, materials, theta0=60.) - target_spec)
    d_insert = np.zeros(d.shape[0] * insert_search_pts)
    # 画插入梯度图
    # layer_number = d.shape[0]
    # for i in range(d_insert.shape[0]):
    #     d_insert[i] = d[0:int(i / insert_search_pts)].sum() + d[int(
    #         i / insert_search_pts)] / insert_search_pts * (i % insert_search_pts)
    #     i += 1
    # plt.plot(d_insert, insert_gradient, label='insert gradient')
    # plt.xlabel('d/nm')
    # plt.legend()
    # plt.title(f'insert gradient, layer num={layer_number}')
    # plt.show()
    # 找最小梯度的地方插入一层
    insert_index = np.argmin(insert_gradient)
    insert_layer_num = int(insert_index / insert_search_pts) + 1
    inserted_layer_thickness = d[insert_layer_num - 1]
    insert_position = (insert_index % insert_search_pts) * d[insert_layer_num - 1] / insert_search_pts
    insert_thickness = 1e-3
    # 这个layer_num是数组的index
    # if materials[insert_layer_num - 1] == 'SiO2_OIC':
    #     insert_material = 'Nb2O5_OIC'
    #     inserted_material = 'SiO2_OIC'
    # elif materials[insert_layer_num - 1] == 'Nb2O5_OIC':
    #     insert_material = 'SiO2_OIC'
    #     inserted_material = 'Nb2O5_OIC'
    # d = np.insert(d, insert_layer_num - 1, insert_position)
    # d = np.insert(d, insert_layer_num, insert_thickness)
    # d[insert_layer_num + 1] = inserted_layer_thickness - insert_position
    # materials = np.insert(materials, insert_layer_num - 1, inserted_material)
    # materials = np.insert(materials, insert_layer_num, insert_material)
    d, materials = inserted_layers(d, materials, insert_layer_num - 1, insert_position, insert_thickness)
    print(f'new layer inserted at {insert_layer_num}th layer, {insert_position}nm')

    return d, materials, insert_layer_num


def calculate_merit(d, materials, wls, target_spec):
    assert d.shape[0] == materials.shape[0], 'shape of d and materials don\'t align'
    f = get_spectrum(wls, d, materials, theta0=60.) - target_spec
    merit = np.sqrt((f ** 2 / wls.shape[0]).sum())
    return merit


def calculate_optical_thickness(d, materials, wl=750):
    optical_thickness = np.dot(d, get_n(wl, materials)[1:-1, 0].real)
    return optical_thickness


class InitializeParameters:
    def __init__(self, ratio=1):
        self.initial_optical_thickness = None
        self.target_optical_thickness = None
        self.optical_thickness_ratio = ratio
        self.available_materials = np.array(['SiO2', 'TiO2'])
        self.test_wl = 750
        self.ratio = ratio

    def init_target_layers(self, do_plot_target_spec=False):
        # Generate target layer structure
        np.random.seed(1)
        # d_target = np.random.random(50) * 200
        # print(f'd_target.sum = {d_target.sum()}')
        d_target = np.array([123, 456, 789, 234, 567], dtype='double')
        materials_target = np.array([])
        for i in range(d_target.shape[0]):
            materials_target = np.append(materials_target, self.available_materials[i % 2])
        # Target optical thickness
        self.target_optical_thickness = calculate_optical_thickness(d_target, materials_target, self.test_wl)
        # Generate and plot target spectrum (R)
        wls = np.linspace(500, 1000, 500)
        target_spec = get_spectrum(wls, d_target, materials_target, theta0=60.)
        if do_plot_target_spec:
            plt.plot(wls, target_spec[0:wls.shape[0]], label='target spectrum')
            plt.legend()
            plt.xlabel('wl/nm')
            plt.show()
        return d_target, materials_target, wls, target_spec

    def init_init_layers(self):
        # Generate initial layer structure
        assert self.target_optical_thickness is not None, 'target not genereated'
        test_thickness = self.target_optical_thickness
        d = np.array([test_thickness], dtype='double')
        materials = np.array([])
        available_materials = np.array(['SiO2', 'TiO2'])
        for i in range(d.shape[0]):
            materials = np.append(materials, available_materials[i % 2])
        # In this case the initial structure has only 1 layer, so simply scale the thickness to desired
        test_optical_thickness = calculate_optical_thickness(d, materials, self.test_wl)
        initial_thickness = self.target_optical_thickness * self.ratio * (test_thickness / test_optical_thickness)
        d[0] = initial_thickness
        self.initial_optical_thickness = initial_thickness * (test_optical_thickness / test_thickness)
        return d, materials


def main():
    ratios = np.linspace(0.5, 2, 15)
    merit_after_50_iters = np.array([])
    for ratio in ratios:
        # Ratio is thickness ratio! In the initialization functions optical thickness will be calculated
        # Initialize target and initial structure
        current_initialize = InitializeParameters(ratio)
        d_target, materials_target, wls, target_spec = current_initialize.init_target_layers(True)
        d_init, materials_init = current_initialize.init_init_layers()
        d = d_init.copy()
        materials = materials_init.copy()

        # Print initial layer structure
        print(np.hstack((d, materials)))
        # Initialize optimization parameters
        insert_layer_num = 0
        insert_count = -1
        merit = calculate_merit(d, materials, wls, target_spec)
        layer_number = d.shape[0]
        # Initialize training process record function
        descent_process_this = descent_process(layer_number, merit, optical_thickness_ratio=current_initialize.
                                               initial_optical_thickness / current_initialize.target_optical_thickness)

        for insert_count in range(50):
            # 梯度下降
            d, materials, insert_layer_num, merit = gradient_descent(wls, target_spec, d, materials, insert_layer_num)

            layer_number = d.shape[0]
            # last_insert_layer_thickness = d[insert_layer_num]
            # plot_layer_thickness(d, materials, insert_layer_num, last_insert_layer_thickness)
            # plot_spectrum(wls, target_spec, d, materials, layer_number, merit, insert_count)
            descent_process_this = descent_process_this(layer_number, merit)

            # 插入新层
            insert_search_pts = 100
            d, materials, insert_layer_num = insert_1_layer(wls, target_spec, d, materials, insert_search_pts)

            # np.savetxt(f'generated_data\\5layer_60degree\\count{insert_count+1}_layer{layer_number}_merit{merit}.txt', d)
            # np.savetxt(f'generated_data\\5layer_60degree\\count{insert_count+1}_materials.txt', materials, fmt='%s')\

        print(merit_after_50_iters)
        merit_after_50_iters = np.append(merit_after_50_iters, merit)
        np.savetxt(f'../results/training_plot_different_init_thickness/merit_after_50_insertions_ratio{ratio}.txt',
                   merit_after_50_iters)
        np.savetxt('../results/training_plot_different_init_thickness/optic_thickness_ratios.txt', ratios)


if __name__ == '__main__':
    main()
