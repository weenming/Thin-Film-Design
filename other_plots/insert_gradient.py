import numpy as np
from get_jacobi import get_jacobi
from get_spectrum import get_spectrum
from get_n import get_n
import matplotlib.pyplot as plt


def plot_layer_thickness(d, materials, insert_layer_num=-1, last_insert_layer_thickness=0):
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
            if i == 0:
                insert_material = materials[i+1]
            else:
                insert_material = materials[i-1]
            materials_new = np.insert(materials, i, materials[i])
            materials_new = np.insert(materials_new, i+1, insert_material)
            d_new = np.insert(d, i, d[i]*j/insert_search_pts)
            d_new[i+1] *= 1 - j/insert_search_pts
            d_new = np.insert(d_new, i+1, 0.001)
            insert_jacobi[:, i*insert_search_pts+j:i*insert_search_pts+j+1] = (get_spectrum(wls, d_new, materials_new) - get_spectrum(wls, d, materials))*1e3

    return insert_jacobi


def gradient_descent(d):
    """
    Carries out the gradient descent process
    :param d:
    :return:
    """
    # 梯度下降
    step_init = 1
    step = step_init
    layer_number = d.shape[0]
    for iter_count in range(1000):
        J = get_jacobi(wls, d, materials)[:, 0:layer_number]
        f = get_spectrum(wls, d, materials) - target_spec
        # 验证梯度
        # if iter_count == 0:
        #     f_diff = get_spectrum(wls, d + 1e-3, materials)
        #     jacobi_diff = f_diff - get_spectrum(wls, d, materials)
        #     print(f'diff : {jacobi_diff*1e3}')
        #     print(f'derivative: {J}')
        delta_d = np.dot(J.T, f)
        d_temp = d - delta_d[:, 0] * step
        # 删掉负的
        for layer_test in range(layer_number):
            if d_temp[layer_test] < 0:
                d_temp[layer_test-1] += d_temp[layer_test+1]
                d_temp = np.delete(d, [layer_test, layer_test+1])
                layer_number = d.shape[0]
                step = step_init
                print(f'negative thickness! layer {layer_test} deleted')
                break
        merit = np.sqrt(((get_spectrum(wls, d_temp, materials) - target_spec) ** 2 / wls.shape[0]).sum())
        if merit < (f ** 2).sum():
            d = d_temp.copy()
            step *= 2
            print(f'\naccepted, merit function{merit}')
            print(d)
            print((delta_d ** 2).sum())
        else:
            step /= 2
            print('\nrejected')
        if step < 1e-15:
            break
    return d


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
    insert_thickness = 1
    # 这个layer_num是数组的index
    if materials[insert_layer_num - 1] == 'SiO2':
        insert_material = 'TiO2'
        inserted_material = 'SiO2'
    elif materials[insert_layer_num - 1] == 'TiO2':
        insert_material = 'SiO2'
        inserted_material = 'TiO2'
    d = np.insert(d, insert_layer_num - 1, insert_position)
    d = np.insert(d, insert_layer_num, insert_thickness)
    d[insert_layer_num + 1] = inserted_layer_thickness - insert_position
    materials = np.insert(materials, insert_layer_num - 1, inserted_material)
    materials = np.insert(materials, insert_layer_num, insert_material)
    print(f'new layer inserted at {insert_layer_num}th layer, {insert_position}nm')

    return d, materials, insert_layer_num


# 先生成目标的光谱：总共3层
x = 800
d_init = np.array([200,0,0], dtype='double')
materials_init = np.array(['SiO2', 'TiO2', 'SiO2'])
wls = np.linspace(500, 1000, 500)

target_spec = get_spectrum(wls, d_init, materials_init)

# plt.plot(wls, target_spec[0:wls.shape[0]], label='target spectrum')
# plt.legend()
# plt.xlabel('wl/nm')
# plt.show()
#
materials = np.array(['SiO2', 'TiO2', 'SiO2', 'TiO2'])
d = np.array([1000, 0, 300, 500], dtype='double')
d = gradient_descent(d)
print(d)


# insert_search_pts = 100
# insert_gradient = np.dot(get_insert_jacobi(wls, d, materials, insert_search_pts).T, get_spectrum(wls, d, materials) - target_spec)
# d_insert = np.zeros(d.shape[0] * insert_search_pts)
# # 画插入梯度图
# layer_number = d.shape[0]
# for i in range(d_insert.shape[0]):
#     d_insert[i] = d[0:int(i / insert_search_pts)].sum() + d[int(
#         i / insert_search_pts)] / insert_search_pts * (i % insert_search_pts)
#     i += 1
# plt.plot(d_insert, insert_gradient, label='insert gradient')
# plt.plot([800, 800], [-0.1, 0.1], ls=':')
# plt.xlabel('d/nm')
# plt.legend()
# plt.title(f'insert gradient demo, layer num={layer_number}')
# plt.show()