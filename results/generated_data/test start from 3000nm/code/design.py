import numpy as np
from get_jacobi import get_jacobi
from get_spectrum import get_spectrum
from get_n import get_n
import matplotlib.pyplot as plt
import pandas as pd
import time


def plot_layer_thickness(d, materials, insert_layer_num=-1, last_insert_layer_thickness=0):
    d_plot = 0
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


def plot_spectrum():
    # 优化到这时候的结果
    print(f'insert_count={insert_count}, d={d}')
    fig_spec, (axR, axT) = plt.subplots(1, 2, sharey=True)
    axR.plot(wls, get_spectrum(wls, d, materials)[0:wls.shape[0]], label='fit')
    axR.plot(wls, target_spec[0:wls.shape[0]], label='target')
    axR.legend()
    axR.set_xlabel('wl/nm')
    axR.set_title(f'R')

    axT = plt.subplot(122)
    axT.plot(wls, get_spectrum(wls, d, materials)[wls.shape[0]:], label='fit')
    axT.plot(wls, target_spec[wls.shape[0]:], label='target')
    axT.legend()
    axT.set_xlabel('wl/nm')
    axT.set_title('T')

    fig_spec.suptitle(f'{layer_number} layers, merit func = {merit}')
    fig_spec.show()


def plot_descent_process(merits=np.array([]), layer_numbers=np.array([])):
    # 损失函数和层数的关系
    merits = np.append(merits, merit)
    layer_numbers = np.append(layer_numbers, layer_number)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(insert_count+2), np.log(merits), label='merits', color='orange')
    ax2.plot(range(insert_count+2), layer_numbers, label='layer number', color='steelblue')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    fig.show()

    def plot_next_descent_process():
        return plot_descent_process(merits, layer_numbers)
    return plot_next_descent_process


def get_insert_jacobi(wls, d, materials, insert_search_pts):
    """
    Calculates jacobi matrix of insertion gradient
    :param wls:
    :param d:
    :param materials:
    :param insert_search_pts:
    :return:
    """
    layer_number = d.shape[0]
    insert_jacobi = np.zeros((2*wls.shape[0], layer_number * insert_search_pts))
    print('start searching for insertion place')
    for i in range(layer_number):
        for j in range(insert_search_pts):
            # print(f'{(i*insert_search_pts+j)/(layer_number*insert_search_pts)} completed')
            if materials[i] == 'SiO2_OIC':
                insert_material = 'Nb2O5_OIC'
            else:
                insert_material = 'SiO2_OIC'

            materials_new = np.insert(materials, i, materials[i])
            materials_new = np.insert(materials_new, i+1, insert_material)
            d_new = np.insert(d, i, d[i]*j/insert_search_pts)
            d_new[i+1] *= 1 - j/insert_search_pts
            d_new = np.insert(d_new, i+1, 0.001)
            insert_jacobi[:, i*insert_search_pts+j:i*insert_search_pts+j+1] = (get_spectrum(wls, d_new, materials_new) - get_spectrum(wls, d, materials))*1e3

    return insert_jacobi


def gradient_descent(d, materials, insert_layer_num):
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

    J = get_jacobi(wls, d, materials)[:, 0:layer_number]
    f = get_spectrum(wls, d, materials) - target_spec
    g = np.dot(J.T, f)
    A = np.dot(J.T, J)
    nu = 2
    mu = 1
    for gradient_descent_count in range(10000):
        # 验证梯度
        # if iter_count == 0:
        #     f_diff = get_spectrum(wls, d + 1e-3, materials)
        #     jacobi_diff = f_diff - get_spectrum(wls, d, materials)
        #     print(f'diff : {jacobi_diff*1e3}')
        #     print(f'derivative: {J}')

        # LM算法，下降
        J = get_jacobi(wls, d, materials)[:, 0:layer_number]
        f = get_spectrum(wls, d, materials) - target_spec
        h = np.dot(np.linalg.inv(A + mu * np.identity(layer_number)), -g)
        d_new = d + h[:, 0]
        # spectrum 是 2N by 1 向量

        F_d = (f ** 2).sum()
        # spectrum 是 6N by 1 向量
        spectrum_new = get_spectrum(wls, d_new, materials)

        f_new = spectrum_new - target_spec
        F_dnew = (f_new ** 2).sum()
        rho = (F_d - F_dnew) / np.dot(h.T, mu * h - g).item()
        if rho > 0:
            d = d_new.copy()
            f = f_new
            g = np.dot(J.T, f)
            A = np.dot(J.T, J)
            mu = mu * np.amax(np.array([1 / 3, 1 - (2 * rho - 1) ** 3])).item()
            nu = 2
            merit = np.sqrt((f ** 2 / wls.shape[0]).sum())
            print(f'\naccepted, merit function{merit}')
            # print(d)
        else:
            print('\ndescent rejected')
            mu = mu * nu
            nu = 2 * nu

        # 删掉负的
        for layer_test in range(layer_number):
            if d[layer_test] < 0:
                if layer_test != layer_number-1 and layer_test != 0:
                    d[layer_test - 1] += d[layer_test + 1]
                    d = np.delete(d, [layer_test, layer_test + 1])
                    materials = np.delete(materials, [layer_test, layer_test + 1])
                else:
                    d = np.delete(d, layer_test)
                    materials = np.delete(materials, layer_test)

                layer_number = d.shape[0]
                J = get_jacobi(wls, d, materials)[:, 0:layer_number]
                f = get_spectrum(wls, d, materials) - target_spec
                g = np.dot(J.T, f)
                A = np.dot(J.T, J)
                nu = 2
                mu = 1


                insert_layer_num=-1
                print(f'negative thickness! layer {layer_test} deleted')
                break

        if np.amax(abs(h)).item() < 1e-11:
            merit = np.sqrt(((get_spectrum(wls, d, materials) - target_spec) ** 2 / wls.shape[0]).sum())
            print(f'\noptimization of {layer_number}layer completed, merit function={merit}')
            break
    return d, materials, insert_layer_num, merit


def insert_1_layer(d, materials, insert_search_pts):
    """
    find the layer and position to insert the new layer
    :param d:
    :param materials:
    :param insert_search_pts:
    :return: inserted d, materials and the index of the inserted layer
    """
    insert_gradient = np.dot(get_insert_jacobi(wls, d, materials, insert_search_pts).T,
                             get_spectrum(wls, d, materials) - target_spec)
    d_insert = np.zeros(d.shape[0] * insert_search_pts)
    # 画插入梯度图
    layer_number = d.shape[0]
    for i in range(d_insert.shape[0]):
        d_insert[i] = d[0:int(i / insert_search_pts)].sum() + d[int(
            i / insert_search_pts)] / insert_search_pts * (i % insert_search_pts)
        i += 1
    plt.plot(d_insert, insert_gradient, label='insert gradient')
    plt.xlabel('d/nm')
    plt.legend()
    plt.title(f'insert gradient, layer num={layer_number}')
    plt.show()
    # 找最小梯度的地方插入一层
    insert_index = np.argmin(insert_gradient)
    insert_layer_num = int(insert_index / insert_search_pts) + 1
    inserted_layer_thickness = d[insert_layer_num - 1]
    insert_position = (insert_index % insert_search_pts) * d[insert_layer_num - 1] / insert_search_pts
    insert_thickness = 1e-3
    # 这个layer_num是数组的index
    if materials[insert_layer_num - 1] == 'SiO2_OIC':
        insert_material = 'Nb2O5_OIC'
        inserted_material = 'SiO2_OIC'
    elif materials[insert_layer_num - 1] == 'Nb2O5_OIC':
        insert_material = 'SiO2_OIC'
        inserted_material = 'Nb2O5_OIC'
    d = np.insert(d, insert_layer_num - 1, insert_position)
    d = np.insert(d, insert_layer_num, insert_thickness)
    d[insert_layer_num + 1] = inserted_layer_thickness - insert_position
    materials = np.insert(materials, insert_layer_num - 1, inserted_material)
    materials = np.insert(materials, insert_layer_num, insert_material)
    print(f'new layer inserted at {insert_layer_num}th layer, {insert_position}nm')

    return d, materials, insert_layer_num


def calculate_merit(d, materials, wls, target_spec):
    assert d.shape[0] == materials.shape[0], 'shape of d and materials don\'t align'
    f = get_spectrum(wls, d, materials) - target_spec
    merit = np.sqrt((f ** 2 / wls.shape[0]).sum())
    return merit
# 先生成目标的光谱：总共3层
x = 1000
d_init = np.array([x, 200, 2000-x], dtype='double')
materials_init = np.array(['SiO2', 'TiO2', 'SiO2'])
wls = np.linspace(500, 1000, 500)
target_spec = get_spectrum(wls, d_init, materials_init)


plt.plot(wls, target_spec[0:wls.shape[0]], label='target spectrum')
plt.legend()
plt.xlabel('wl/nm')
plt.show()


# 然后开始优化(3层)，看看不同的初始厚度
# d = np.array([6.85230688e+01, 9.67156939e+00, 6.56463217e+01, 1.96907787e+02,
#               2.31305921e+02, 2.80614152e+00, 5.43837830e+01, 2.29264939e+02,
#               4.15289899e+01, 2.16215359e+02, 3.40564809e+02, 4.90165010e+01,
#               5.64641636e+01, 2.54690864e+02, 3.81007708e+01, 4.06659094e+01,
#               2.43764330e+02, 1.00000000e-03, 6.87540419e+01, 4.23146921e+02,
#               4.98807700e+01, 3.39487152e+01, 3.07140185e+02, 3.86527922e+01,
#               3.53278207e+01], dtype='double')

# OIC A1, 插入在第一层。。。
# d = np.array([0.00000000e+00, 1.00000000e-03, 9.04248844e+01, 1.59707163e+02,
#               2.84118407e+02, 1.73221763e+01, 2.75895871e+01, 2.43269466e+02,
#               2.28656956e+01, 2.42649517e+01, 4.59775859e+01, 1.02776076e+01,
#               7.39225662e+02, 7.62007993e+00, 4.94567636e+01, 1.96276271e+01,
#               3.03807701e+01, 3.12259360e+01, 1.25832851e+01, 9.50888237e+02])

# d = np.loadtxt('count2_layer25_merit0.06436844231397111.txt')
d = np.array([3000], dtype='double')
materials = np.array([])
available_materials = np.array(['SiO2_OIC', 'Nb2O5_OIC'])
for i in range(d.shape[0]):
    materials = np.append(materials, available_materials[i % 2])
print(np.hstack((d, materials)))

insert_layer_num = 0

# 初始化merit和层数
insert_count = -1
merit = calculate_merit(d, materials, wls, target_spec)
layer_number = d.shape[0]
plot_descent_process_this = plot_descent_process()

for insert_count in range(100):
    layer_number = d.shape[0]
    # 梯度下降
    d, materials, insert_layer_num, merit = gradient_descent(d, materials, insert_layer_num)
    layer_number = d.shape[0]
    last_insert_layer_thickness = d[insert_layer_num]
    # 画结束梯度下降的厚度图
    plot_layer_thickness(d, materials, insert_layer_num, last_insert_layer_thickness)
    # 画光谱
    plot_spectrum()
    # 画merit和层数关系
    plot_descent_process_this = plot_descent_process_this()

    # 插入新层
    insert_search_pts = 100
    d, materials, insert_layer_num = insert_1_layer(d, materials, insert_search_pts)

    np.savetxt(f'generated_data\\count{insert_count+1}_layer{layer_number}_merit{merit}.txt', d)
    np.savetxt(f'generated_data\\count{insert_count+1}_materials.txt', materials, fmt='%s')





# 优化一层厚度的损失函数分布
# plot = np.zeros((5000, 2))
# for i in range(5000):
#     d = np.array([i])
#     # f = get_spectrum(wls, d + 1e-3, materials) - get_spectrum(wls, d, materials)
#     # gradient = np.dot(f.T, get_spectrum(wls, d, materials) - target_spec)
#     plot[i, :] = np.array([d[0], ((get_spectrum(wls, d, materials) - target_spec)**2).sum()])
#     print(plot[i, :])
# plt.plot(plot[:, 0], plot[:, 1])
# plt.xlabel('d/nm')
# plt.show()
