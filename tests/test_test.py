import numpy as np
from get_n import get_n
import matplotlib.pyplot as plt
import design
import get_insert_jacobi
import time


# insert_search_pts = 100
# insert_index = 299
# d = np.array([10, 2, 30], dtype='double')
# materials = np.array(['SiO2', 'TiO2', 'SiO2'])


# insert_layer_num = int(insert_index / insert_search_pts) + 1
# inserted_layer_thickness = d[insert_layer_num-1]
# insert_position = (insert_index % insert_search_pts) * d[insert_layer_num-1] / insert_search_pts
# insert_thickness = 1
# # 这个layer_num是数组的index
# if materials[insert_layer_num - 1] == 'SiO2':
#     insert_material = 'TiO2'
#     inserted_material = 'SiO2'
# elif materials[insert_layer_num - 1] == 'TiO2':
#     insert_material = 'SiO2'
#     inserted_material = 'TiO2'
# d = np.insert(d, insert_layer_num-1, insert_position)
# d = np.insert(d, insert_layer_num, insert_thickness)
# d[insert_layer_num+1] = inserted_layer_thickness - insert_position
# materials = np.insert(materials, insert_layer_num-1, inserted_material)
# materials = np.insert(materials, insert_layer_num, insert_material)
#
# print(d)
# print(materials)

# layer_number = d.shape[0]
# insert_layer_num = 2
# last_insert_layer_thickness = d[insert_layer_num]
# d_plot = 0
# for i in range(layer_number):
#     # 以600nm的折射率表示
#     n_plot = get_n(600, materials)[i + 1].real
#     if i == insert_layer_num:
#         plt.plot([d_plot, d_plot + d[i]], [n_plot, n_plot], color='red',
#                  label=f'thickness={last_insert_layer_thickness}')
#     else:
#         plt.plot([d_plot, d_plot + d[i]], [n_plot, n_plot], color='blue')
#     d_plot += d[i]
# plt.show()

# plt.scatter([1, 2,4,6,8,10,12,14,16, 18], [0.07987050797316056, 0.05986496668286043, 0.02393618455693054,0.01012338019496746, 0.007543112031073434, 0.00475655463049579, 0.003624386109712861, 0.0034357137990625917, 0.0030, 0.0005])
# plt.show()
# merits=np.array([0.1, 0.02, 0.00003])
# insert_count = 2
# layer_numbers = np.array([1,2,3])
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(range(insert_count + 1), -np.log(merits), label='merits', color='orange')
# ax2.plot(range(insert_count + 1), layer_numbers, label='layer number', color='steelblue')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper left')
#
# fig.show()
# d = np.array([1, 1, 1], dtype='double')
# available_materials = np.array(['SiO2_OIC', 'Nb2O5_OIC'])
# materials=np.array([])
# for i in range(d.shape[0]):
#     materials = np.append(materials, available_materials[i % 2])
#
# print(design.inserted_layers(d, materials, 0, 0.))

d = np.zeros(400,dtype='double')+1000
materials = np.array([])
available_materials = np.array(['SiO2_OIC', 'Nb2O5_OIC'])
for i in range(d.shape[0]):
    materials = np.append(materials, available_materials[i % 2])

print('faster:')
t1 = time.time()
get_insert_jacobi.get_insert_jacobi_faster(np.linspace(500, 700, 10), d, materials, 20)
t2 = time.time()
print(t2-t1)
print('current:')
t1 = time.time()
design.get_insert_jacobi(np.linspace(500, 700, 10), d, materials, 20)
t2 = time.time()
print(t2-t1)
