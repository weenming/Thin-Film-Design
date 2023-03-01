import numpy as np
import film


def inserted_layers(film1: film.Film, insert_layer_num, insert_position, insert_thickness=0.0000):
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
        get_insert_jacobi.get_insert_jacobi_faster(
            wls, d, materials, insert_search_pts, theta0=60.).T,
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
    insert_position = (insert_index % insert_search_pts) * \
        d[insert_layer_num - 1] / insert_search_pts
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
    d, materials = inserted_layers(
        d, materials, insert_layer_num - 1, insert_position, insert_thickness)
    print(
        f'new layer inserted at {insert_layer_num}th layer, {insert_position}nm')

    return d, materials, insert_layer_num
