from numpy import *
import csv
from get_jacobi import get_jacobi
from get_spectrum import get_spectrum
import matplotlib.pyplot as plt
import pandas as pd
import time

# 这个程序用来计算OIC 2022 问题A1。
# 用LM法优化，使光谱接近目标光谱
def main():
    # 读取目标光谱
    target_spectrum_3column = loadtxt(open("C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\oic2022-PA-Measurements.csv", "rb"), delimiter=",", skiprows=1)
    # 合并R和T光谱
    target_spectrum = vstack(
        (array([target_spectrum_3column[0:, 2]]).T, array([target_spectrum_3column[0:, 1]]).T)) / 100  # 百分制变成小数
    print(target_spectrum)
    wls = array([i for i in range(220, 1701)])
    wls = array([wls]).T

    # 不算基底(膜数=layer_number)，并假设最上面一层是Nb2O5
    layer_number = 35
    d = zeros((layer_number, 1), dtype='double') + 50

    # n_indices 第一个是折射率变化里边的m，第二个是吸收的系数，0代表没吸收,没有基底。n_indices长度是膜层数(不算基底)
    n_indices = zeros((layer_number, 2), dtype=int)
    for i in range(layer_number):
        if i % 2 != 0:
            # Nb2O5
            n_indices[i] = [3, 1]
        else:
            # SiO2
            n_indices[i] = [3, 0]
    # 迭代，梯度下降
    gradient_scaler = 10
    iter_count = 0
    # 第一次算
    # 用差分算jacobi
    delta = 1e-7
    jacobi_d = zeros((2 * wls.shape[0],  d.shape[0]))
    for i in range(d.shape[0]):
        d_delta = d.copy()
        d_delta[i, 0] = d[i, 0] + delta
        jacobi_d[0:, i] = (get_spectrum(wls, d_delta, n_indices).T - get_spectrum(wls, d, n_indices).T).real / delta
        # print((get_spectrum_diff(wls, d_delta, n_indices).T - get_spectrum_diff(wls, d, n_indices).T) / delta)
    spectrum = get_spectrum(wls, d, n_indices)
    f = spectrum - target_spectrum
    g = dot(jacobi_d.T, f)
    A = dot(jacobi_d.T, jacobi_d)
    nu = 2
    mu = 1
    h = dot(linalg.inv(A + mu * identity(layer_number)), -g)
    while 1:
        start = time.perf_counter()
        layer_number = d.shape[0]
        iter_count += 1
        # jacobi是2N by ~3M 矩阵 行按照R1, R2... Tn排列; 列按照d1, d2, ...dm, n1,n2,...k1,...,kn排列,并注意到只能更新奇数层的k
        # 用差分算jacobi
        delta = 1e-7
        jacobi_d = zeros((2 * wls.shape[0],  d.shape[0]))
        for i in range(d.shape[0]):
            d_delta = d.copy()
            d_delta[i, 0] = d[i, 0] + delta
            jacobi_d[0:, i] = (get_spectrum(wls, d_delta, n_indices).T - get_spectrum(wls, d, n_indices).T).real / delta
            # print((get_spectrum_diff(wls, d_delta, n_indices).T - get_spectrum_diff(wls, d, n_indices).T) / delta)

        # spectrum 是 2N by 1 列向量
        spectrum = get_spectrum(wls, d, n_indices)
        # 用LM算法更新
        h = dot(linalg.inv(A + mu * identity(layer_number)), -g)
        d_new = d + h
        F_d = ((get_spectrum(wls,d,n_indices) - target_spectrum) * (get_spectrum(wls,d,n_indices) - target_spectrum)).sum()
        F_dnew = ((get_spectrum(wls,d_new,n_indices) - target_spectrum) * (get_spectrum(wls,d_new,n_indices) - target_spectrum)).sum()
        rho = (F_d-F_dnew)/ dot(h.T, mu*h-g).item()
        if rho > 0:
            d = d_new.copy()
            f = spectrum - target_spectrum
            g = dot(jacobi_d.T, f)
            A = dot(jacobi_d.T, jacobi_d)
            mu = mu * amax(array([1/3, 1-(2 * rho - 1) ** 3])).item()
            nu = 2
        else:
            mu = mu * nu
            nu = 2* nu

        # 负厚度,如果不是最后一层就删掉

        if d[d.shape[0]-1,0]<0:
            d[d.shape[0]-1,0] = 0
            print('negative thickness!')
        i = 0
        while i<layer_number-1:
            if d[i, 0] < 0:
                d[i - 1, 0] += d[i + 1, 0]
                d = delete(d, [i, i + 1], axis=0)
                layer_number = d.shape[0]
                n_indices = delete(n_indices, [i, i + 1], axis=0)
                print('negative thickness! layer deleted')
                # 更新jacobi
                F_d = ((get_spectrum(wls, d, n_indices) - target_spectrum) * (
                            get_spectrum(wls, d, n_indices) - target_spectrum)).sum()
                F_dnew = ((get_spectrum(wls, d_new, n_indices) - target_spectrum) * (
                            get_spectrum(wls, d_new, n_indices) - target_spectrum)).sum()
                rho = (F_d - F_dnew) / dot(h.T, mu * h - g).item()
                f = spectrum - target_spectrum
                g = dot(jacobi_d.T, f)
                A = dot(jacobi_d.T, jacobi_d)
            i=i+1

        # # 判断正负来决定改变层的特性:partial/partial n > 0 -> n=n+Delta ; partial/ partial k>0-> 变成更高的吸收
        # jacobi_n = jacobi_diff[0:, layer_number:2*layer_number]
        # jacobi_k = jacobi_diff[0:, 2*layer_number:3*layer_number]
        # partial_n = dot(jacobi_n.T, (spectrum - target_spectrum))
        # partial_k = dot(jacobi_k.T, (spectrum - target_spectrum))
        # # 更新折射率，吸收率的系数
        # for i in range(layer_number):
        #     if partial_n[i] < 0 and n_indices[i, 0] < 5:
        #         n_indices[i, 0] += 1
        #     elif partial_n[i] > 0 and n_indices[i, 0] > 1:
        #         n_indices[i, 0] -= 1
        #     if n_indices[i, 1] != 0:
        #         # 只有Nb2O5有吸收
        #         if partial_k[i] < 0 and n_indices[i, 1] < 3:
        #             n_indices[i, 1] += 1
        #         elif partial_k[i] > 0 and n_indices[i, 1] > 1:
        #             n_indices[i,1] -= 1

        error = sqrt(((spectrum-target_spectrum)*(spectrum-target_spectrum)).sum()/2962)
        end = time.perf_counter()
        # 退出循环的条件
        if error < 0.01 or iter_count > 100:
            break
        print(f'iteration {iter_count} completed, error = {error}')
        print(f'iteration takes: {end-start}s')
    # 存到文件 这个应该是反了
    structure = pd.DataFrame([['(name and affiliation)', ''], ['email address', ''], ['', '']], columns=[1, 2])
    for i in range(n_indices.shape[0]):
        if n_indices[i, 1] == 0:
            # SiO2 Ln(i),d(i)
            n = n_indices[i, 0]
            structure = pd.concat([structure, pd.DataFrame({1: [f'L{n}'], 2: [f'{d[i, 0].real}']})])
        else:
            # Nb2O5 Hn(i)k(i), d(i)
            n = n_indices[i, 0]
            k = n_indices[i, 1]
            structure = pd.concat(
                [structure, pd.DataFrame({1: [f'H{n}{k}'], 2: [f'{d[i, 0].real}']})])
    structure.to_csv(f'C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\result\\initlayernumber_{layer_number}_error_{error}.txt', index=False, header=None, sep=' ')

    # 画图
    points_number = wls.shape[0]
    spec = get_spectrum(wls, d, n_indices)
    savetxt('spec.csv', spec, delimiter=',')
    plt.plot(wls, spec[0: points_number].real)
    plt.plot(wls, spec[points_number: 2 * points_number].real)
    # 是不是差一个系数。。。没吸收的时候T+R应该是1
    plt.plot(wls, 1-spec[0: points_number])
    plt.scatter(wls,target_spectrum[0:wls.shape[0],0])
    plt.scatter(wls,target_spectrum[wls.shape[0]:2*wls.shape[0],0])
    plt.show()


def main_needle():
    # 读取目标光谱，顺序: non-polarized, p-polarized, s-polarized
    target_spectrum_a7 = loadtxt(
        open("C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\oic2022-PA-Measurements.csv",
             "rb"),
        delimiter=",", skiprows=1)/100

    print(target_spectrum_a7.shape)
    target_spectrum = vstack(
        (target_spectrum_a7[0:, 2:3], target_spectrum_a7[0:, 1:2]))
    wls = array([i for i in range(220, 1701)])
    wls = array([wls]).T

    # # # 不算基底(膜数=layer_number)，并假设最上面一层是Nb2O5
    # layer_number = 3
    # d = zeros((layer_number, 1), dtype='double') + 50
    #
    # # n_indices 第一个是折射率变化里边的m，第二个是吸收的系数，0代表没吸收,没有基底。n_indices长度是膜层数(不算基底)
    # n_indices = zeros((layer_number, 2), dtype=int)
    # for i in range(layer_number):
    #     if i % 2 == 0:
    #         # Nb2O5
    #         n_indices[i] = [3, 2]
    #     else:
    #         # SiO2
    #         n_indices[i] = [3, 0]

    # 从文件读取层结构
    path = 'C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\result\\layernumber_27_error_0.015051447475473866_n.txt'
    loaded_structure = loadtxt(path, skiprows=3, dtype=str_)
    print(loaded_structure)
    d = array([loaded_structure[0:, 1].astype(float)]).T
    layer_number = d.shape[0]
    d = flip(d, axis=0)
    print(d)
    n_indices = zeros((d.shape[0], 2),dtype=int)
    for i in range(loaded_structure.shape[0]):
        if len(loaded_structure[i, 0:][0]) == 2:
            n_indices[i, 0:] = [int(loaded_structure[i, 0][1]), 0]
        else:
            n_indices[i, 0:] = [int(loaded_structure[i, 0][1]), int(loaded_structure[i, 0][2])]
    n_indices = flip(n_indices, axis=0)
    print(n_indices)

    # 迭代，lm算法，到达极小时新增一层。
    # 用求导算jacobi, 先算一个初始值
    jacobi_d = get_jacobi(wls, d, n_indices)[0:,0:layer_number]
    # spectrum 是 2N by 1 向量
    spectrum = get_spectrum(wls, d, n_indices)
    f = spectrum - target_spectrum
    g = dot(jacobi_d.T, f)
    A = dot(jacobi_d.T, jacobi_d)
    nu = 2
    mu = 1
    insert_search_points = 1
    optimization_record = []
    while layer_number < 75:
        iter_count = 0
        while 1:
            layer_number = d.shape[0]
            iter_count += 1

            # jacobi是2N by ~3M 矩阵 行按照R1, R2... Tn排列; 列按照d1, d2, ...dm, n1,n2,...k1,...,kn排列,并注意到只能更新奇数层的k
            # LM算法，下降
            jacobi_d = get_jacobi(wls, d, n_indices)[0:, 0:layer_number]
            h = dot(linalg.inv(A + mu * identity(layer_number)), -g)
            d_new = d + h
            # spectrum 是 2N by 1 向量
            spectrum = get_spectrum(wls, d, n_indices)

            f = spectrum - target_spectrum
            F_d = (f ** 2).sum()
            # spectrum 是 6N by 1 向量
            spectrum_new = get_spectrum(wls, d_new, n_indices)

            f_new = spectrum_new - target_spectrum
            F_dnew = (f_new ** 2).sum()
            rho = (F_d - F_dnew) / dot(h.T, mu * h - g).item()
            if rho > 0:
                d = d_new.copy()
                f = f_new
                g = dot(jacobi_d.T, f)
                A = dot(jacobi_d.T, jacobi_d)
                mu = mu * amax(array([1 / 3, 1 - (2 * rho - 1) ** 3])).item()
                nu = 2
            else:
                print('descent rejected')
                mu = mu * nu
                nu = 2 * nu
            # 负厚度,如果不是最后一层就删掉这层并合并相邻2层；如果是最后一层就只删掉自己
            i = 0
            while i < layer_number:
                if d[i, 0] < 0:
                    if i == layer_number-1:
                        d = delete(d, layer_number - 1, axis=0)
                        n_indices = delete(n_indices, layer_number - 1, axis=0)
                        layer_number = d.shape[0]
                        print('negative thickness! last layer deleted')
                    elif i == 0:
                        d = delete(d, 0, axis=0)
                        n_indices = delete(n_indices, 0, axis=0)
                        layer_number = d.shape[0]
                        print('negative thickness! first layer deleted')
                    else:
                        d[i - 1, 0] += d[i + 1, 0]
                        d = delete(d, [i, i + 1], axis=0)
                        n_indices = delete(n_indices, [i, i + 1], axis=0)
                        layer_number = d.shape[0]
                        print(f'negative thickness! layer {i} deleted')
                    # 要更新一下梯度下降的参数
                    # 用求导算jacobi, 先算一个初始值
                    jacobi_d = get_jacobi(wls, d, n_indices)[0:, 0:layer_number]
                    # spectrum 是 6N by 1 向量
                    spectrum = get_spectrum(wls, d, n_indices)
                    f = spectrum - target_spectrum
                    g = dot(jacobi_d.T, f)
                    A = dot(jacobi_d.T, jacobi_d)
                    mu = 1
                    nu = 2
                i = i+1
            # # 判断正负来决定改变层的特性:partial/partial n > 0 -> n=n+Delta ; partial/ partial k>0-> 变成更高的吸收
            # jacobi_n = jacobi_diff[0:, layer_number:2*layer_number]
            # jacobi_k = jacobi_diff[0:, 2*layer_number:3*layer_number]
            # partial_n = dot(jacobi_n.T, (spectrum - target_spectrum))
            # partial_k = dot(jacobi_k.T, (spectrum - target_spectrum))
            # # 更新折射率，吸收率的系数
            # for i in range(layer_number):
            #     if partial_n[i] < 0 and n_indices[i, 0] < 5:
            #         n_indices[i, 0] += 1
            #     elif partial_n[i] > 0 and n_indices[i, 0] > 1:
            #         n_indices[i, 0] -= 1
            #     if n_indices[i, 1] != 0:
            #         # 只有Nb2O5有吸收
            #         if partial_k[i] < 0 and n_indices[i, 1] < 3:
            #             n_indices[i, 1] += 1
            #         elif partial_k[i] > 0 and n_indices[i, 1] > 1:
            #             n_indices[i,1] -= 1

            error = sqrt(F_d / 1481 / 2)
            print(f'iteration {iter_count} completed, error = {error}')
            if amax(abs(h)).item() < 1e-11:
                print(f'optimization of {layer_number}layer completed, error={error}')
                break
        append(optimization_record, error)
        savetxt('record.txt', optimization_record)


        start = time.perf_counter()
        # 找到梯度为正的地方，插入一层 (needle optimization)
        # 因为不知道迭代的时候最后一次F还是F_new，重新算一次.
        spectrum = get_spectrum(wls, d, n_indices)
        f = spectrum - target_spectrum
        F_d = (f ** 2).sum()
        # 迭代看哪一层添加的时候最好.
        best_insert_improve = 0
        isinsert = 0
        insert_thickness = 1e-3
        d_temp = []
        n_indices_temp = []
        depth = 0
        scatter_temp = zeros((layer_number * insert_search_points, 2))
        for j in range(0, layer_number):
            if insert_search_points == 100:
                for i in range(insert_search_points):
                    insert_depth = i / insert_search_points * d[j, 0]
                    if n_indices[j, 1] == 0:
                        # SiO2, insert material is Nb2O5
                        n_indices_new = insert(n_indices, j + 1, [3, 0], axis=0)
                        n_indices_new = insert(n_indices_new, j + 1, [3, 2], axis=0)
                    else:
                        n_indices_new = insert(n_indices, j + 1, [3, 2], axis=0)
                        n_indices_new = insert(n_indices_new, j + 1, [3, 0], axis=0)
                    d_new = insert(d, j + 1, d[j, 0] - insert_depth, axis=0)
                    d_new[j, 0] = insert_depth
                    d_new = insert(d_new, j + 1, insert_thickness, axis=0)

                    spectrum_new = get_spectrum(wls, d_new, n_indices_new)
                    f_new = spectrum_new - target_spectrum
                    insert_improve = sqrt((f_new ** 2).sum() / 2962) - sqrt(F_d / 2962)

                    scatter_temp[insert_search_points*j + i, 0:] = [insert_depth + depth, insert_improve]
                    if insert_improve < best_insert_improve:
                        best_insert_improve = insert_improve.copy()
                        d_temp = d_new.copy()
                        n_indices_temp = n_indices_new.copy()
                        insert_layer_index = j
                        layer_number_temp = d_new.shape[0]
                        isinsert = 1
            if insert_search_points == 1:
                insert_depth = 0
                if n_indices[j, 1] == 0:
                    # SiO2, insert material is Nb2O5
                    n_indices_new = insert(n_indices, j + 1, [3, 0], axis=0)
                    n_indices_new = insert(n_indices_new, j + 1, [3, 2], axis=0)
                else:
                    n_indices_new = insert(n_indices, j + 1, [3, 2], axis=0)
                    n_indices_new = insert(n_indices_new, j + 1, [3, 0], axis=0)
                d_new = insert(d, j + 1, d[j, 0], axis=0)
                d_new[j, 0] = 0
                d_new = insert(d_new, j + 1, insert_thickness, axis=0)

                spectrum_new = get_spectrum(wls, d_new, n_indices_new)
                f_new = spectrum_new - target_spectrum
                insert_improve1 = sqrt((f_new ** 2).sum() / 2962) - sqrt(F_d / 2962)

                d_new[j, 0] += insert_thickness
                d_new[j+1, 0] -= insert_thickness
                spectrum_new = get_spectrum(wls, d_new, n_indices_new)
                f_new = spectrum_new - target_spectrum
                insert_improve2 = sqrt((f_new ** 2).sum() / 2962) - sqrt(F_d / 2962)

                d_new[j + 1, 0] += insert_thickness

                insert_improve = (insert_improve1 + insert_improve2) / 2
                scatter_temp[insert_search_points * j, 0:] = [insert_depth + depth, insert_improve]
                if insert_improve < best_insert_improve and insert_improve1 < 0 and insert_improve2 < 0:
                    # 插入两层的话，两层梯度都要负的
                    best_insert_improve = insert_improve.copy()
                    d_temp = d_new.copy()
                    n_indices_temp = n_indices_new.copy()
                    insert_layer_index = j
                    layer_number_temp = d_new.shape[0]
                    isinsert = 1

                # # 验证
                # if j == 1:
                #     print(f'error {sqrt(F_d/2962)}')
                #     print(f'depth{i}, gradient {insert_improve/insert_thickness}')
                #     # print(f'gradient by analytic at {insert_thickness} {1/(2*sqrt(2962)*sqrt((f_new ** 2).sum()))*2*dot(get_jacobi(wls, d_new, n_indices_new)[0:,0:layer_number+2].T, f_new)[j+1, 0].item()}')
                #     # d_gradienttemp = d_new.copy()
                #     # d_gradienttemp[j+1, 0] = 0
                #
                #     # print(f'gradient by analytic at 0 {1/(2*sqrt(2962)*sqrt(F_d))*2 * dot(get_jacobi(wls, d_gradienttemp[0:,0:], n_indices_new)[0:, 0:layer_number + 2].T, f)[j + 1, 0].item()}\n')
            depth += d[j, 0]
        # plt.show()
        end = time.perf_counter()
        print(f'searching for insertion position takes{end - start}')
        # 更新参数
        if isinsert == 1:
            d = d_temp.copy()
            n_indices = n_indices_temp.copy()
            layer_number = layer_number_temp
            if insert_search_points == 1:
                print(f'new layer inserted between {insert_layer_index}th and {insert_layer_index + 1}th layer. gradient is {best_insert_improve/insert_thickness}\n d = {d.T}')
            else:
                print(f'new layer inserted at {insert_layer_index}th layer, gradient is {best_insert_improve/insert_thickness}\n d = {d.T}')
            # 因为添加了层，要更新一下梯度下降的参数
            # 用求导算jacobi, 先算一个初始值
            jacobi_d = get_jacobi(wls, d, n_indices)[0:,0:layer_number]
            # spectrum 是 2N by 1 向量
            spectrum = get_spectrum(wls, d, n_indices)
            f = spectrum - target_spectrum
            g = dot(jacobi_d.T, f)
            A = dot(jacobi_d.T, jacobi_d)
            h_temp = dot(linalg.inv(A + mu * identity(layer_number)), -g)
            mu = 1
            nu = 2
            insert_search_points = 1
            # # 然后先只开放新增的层，到最优之后再开放其他的
            # while 1:
            #     # 差分算梯度
            #     d_test = d.copy()
            #     d_test[insert_index, 0] += insert_thickness / 10
            #     spectrum_test = get_spectrum(wls, d_test, n_indices)
            #     f_test = spectrum_test - target_spectrum
            #     gradient = (f_test**2).sum() - (f**2).sum()
            #
            #     d_temp = d.copy()
            #     steplength = insert_thickness
            #     d_temp[insert_index, 0] -= gradient * steplength
            #     # 看新的能不能下降
            #     f_new = spectrum_new - target_spectrum
            #     descent = (f_new ** 2).sum() - (f ** 2).sum()
            #     if descent < 0:
            #         d = d_temp
            #         steplength *= 2
            #         print(f'inserted layer updated, loss = {sqrt((f_new ** 2).sum()/2962)}')
            #     else:
            #         steplength /= 2
            #         if steplength < insert_thickness / 100:
            #             print('optimization of the inserted layer completed')
            #             break
        else:
            if insert_search_points == 1:
                # 在层间找不到，只好在层内插入
                insert_search_points = 100
                print('cannot find insertion location, search inside layers')
            else:
                # 已经在层中间插入，换成更小的区间再搜索一次
                insert_search_points *= 3
                print(f'cannot find insertion location, search again in {insert_search_points} points')
        # 存到文件
        structure = pd.DataFrame([['(name and affiliation)', ''], ['email address', ''], ['', '']], columns=[1, 2])
        for i in range(n_indices.shape[0]):
            if n_indices[layer_number - 1 - i, 1] == 0:
                # SiO2 Ln(i),d(i)
                n = n_indices[layer_number - 1 - i, 0]
                structure = pd.concat(
                    [structure, pd.DataFrame({1: [f'L{n}'], 2: [f'{d[layer_number - 1 - i, 0].real}']})])
            else:
                # Nb2O5 Hn(i)k(i), d(i)
                n = int(n_indices[layer_number - 1 - i, 0])
                k = int(n_indices[layer_number - 1 - i, 1])
                structure = pd.concat(
                    [structure, pd.DataFrame({1: [f'H{n}{k}'], 2: [f'{d[layer_number - 1 - i, 0].real}']})])
        structure.to_csv(
            f'C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\result\\layernumber_{layer_number}_error_{error}_needle.txt',
            index=False, header=None, sep=' ')
    # 画图
    points_number = wls.shape[0]
    spec = get_spectrum(wls, d, n_indices)
    savetxt('spec.csv', spec, delimiter=',')
    plt.plot(wls, spec[0: points_number].real)
    plt.scatter(wls, target_spectrum[0:wls.shape[0], 0])
    plt.show()



def main_gradient():
    # 读取目标光谱，顺序: non-polarized, p-polarized, s-polarized
    target_spectrum_a7 = loadtxt(
        open("C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\oic2022-PA-Measurements.csv",
             "rb"),
        delimiter=",", skiprows=1)

    print(target_spectrum_a7.shape)
    target_spectrum = vstack(
        (target_spectrum_a7[0:, 2:3], target_spectrum_a7[0:, 1:2]))/100
    wls = array([i for i in range(220, 1701)])
    wls = array([wls]).T

    # 从文件读取层结构
    # path = 'C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\result\\layernumber_27_error_0.015276059021661825.txt'
    # loaded_structure = loadtxt(path, skiprows=3, dtype=str_)
    # print(loaded_structure)
    # d = array([loaded_structure[0:, 1].astype(float)]).T
    # layer_number = d.shape[0]
    # d = flip(d, axis=0)
    # print(d)
    # n_indices = zeros((d.shape[0], 2),dtype=int)
    # for i in range(loaded_structure.shape[0]):
    #     if len(loaded_structure[i, 0:][0]) == 2:
    #         n_indices[i, 0:] = [int(loaded_structure[i, 0][1]), 0]
    #     else:
    #         n_indices[i, 0:] = [int(loaded_structure[i, 0][1]), int(loaded_structure[i, 0][2])]
    # n_indices = flip(n_indices, axis=0)

    layer_number = 25
    d = random.random((25,1))*100
    n_indices = zeros((25,2))
    for i in range(25):
        if i%2 == 0:
            n_indices[i,:]=[3,0]
        else:
            n_indices[i,:]=[3,1]
    # 迭代，lm算法，到达极小时新增一层。
    # 用求导算jacobi, 先算一个初始值
    jacobi_d = get_jacobi(wls, d, n_indices)[0:, 0:layer_number]
    # spectrum 是 2N by 1 向量
    spectrum = get_spectrum(wls, d, n_indices)
    f = spectrum - target_spectrum
    g = dot(jacobi_d.T, f)
    A = dot(jacobi_d.T, jacobi_d)
    nu = 2
    mu = 1
    layer_adjust_n = 0
    while 1:
        iter_count = 0
        while 1:
            layer_number = d.shape[0]
            iter_count += 1

            # jacobi是2N by ~3M 矩阵 行按照R1, R2... Tn排列; 列按照d1, d2, ...dm, n1,n2,...k1,...,kn排列,并注意到只能更新奇数层的k
            # LM算法，下降
            jacobi = get_jacobi(wls, d, n_indices)
            jacobi_d = jacobi[0:, 0:layer_number]
            h = dot(linalg.inv(A + mu * identity(layer_number)), -g)
            d_new = d + h
            # spectrum 是 2N by 1 向量
            spectrum = get_spectrum(wls, d, n_indices)

            f = spectrum - target_spectrum
            F_d = (f ** 2).sum()
            # spectrum 是 6N by 1 向量
            spectrum_new = get_spectrum(wls, d_new, n_indices)

            f_new = spectrum_new - target_spectrum
            F_dnew = (f_new ** 2).sum()
            rho = (F_d - F_dnew) / dot(h.T, mu * h - g).item()
            if rho > 0:
                d = d_new.copy()
                f = f_new
                g = dot(jacobi_d.T, f)
                A = dot(jacobi_d.T, jacobi_d)
                mu = mu * amax(array([1 / 3, 1 - (2 * rho - 1) ** 3])).item()
                nu = 2
            else:
                mu = mu * nu
                nu = 2 * nu
            # 负厚度,如果不是最后一层就删掉这层并合并相邻2层；如果是最后一层就只删掉自己
            i = 0
            while i < layer_number:
                if d[i, 0] < 0:
                    if i == layer_number - 1:
                        d = delete(d, layer_number - 1, axis=0)
                        n_indices = delete(n_indices, layer_number - 1, axis=0)
                        layer_number = d.shape[0]
                        print('negative thickness! last layer deleted')
                    else:
                        d[i - 1, 0] += d[i + 1, 0]
                        d = delete(d, [i, i + 1], axis=0)
                        n_indices = delete(n_indices, [i, i + 1], axis=0)
                        layer_number = d.shape[0]
                        print(f'negative thickness! layer {i} deleted')
                    # 要更新一下梯度下降的参数
                    # 用求导算jacobi, 先算一个初始值
                    jacobi_d = get_jacobi(wls, d, n_indices)[0:, 0:layer_number]

                    # spectrum 是 6N by 1 向量
                    spectrum = get_spectrum(wls, d, n_indices)

                    f = spectrum - target_spectrum
                    g = dot(jacobi_d.T, f)
                    A = dot(jacobi_d.T, jacobi_d)
                    h = dot(linalg.inv(A + mu * identity(layer_number)), -g)
                    mu = 1
                    nu = 2
                i = i + 1



            error = sqrt(F_d / 1481 / 2)
            print(f'iteration {iter_count} completed, error = {error}')
            if amax(abs(h)).item() < 1e-3 and iter_count > 10:
                print(f'optimization of {layer_number}layer completed, error={error}')
                break

        # 因为不知道迭代的时候最后一次F还是F_new，重新算一次.
        spectrum = get_spectrum(wls, d, n_indices)
        f = spectrum - target_spectrum
        F_d = (f ** 2).sum()
        # 判断正负来决定改变层的特性:partial/partial n > 0 -> n=n+Delta ; partial/ partial k>0-> 变成更高的吸收
        # 干脆直接遍历好了，，，找到最好的那一层。应该算层数次正向传播，几分钟吧
        jacobi_n = jacobi[0:, layer_number:2 * layer_number]
        jacobi_k = jacobi[0:, 2 * layer_number:3 * layer_number]
        partial_n = dot(jacobi_n.T, (spectrum - target_spectrum))
        partial_k = dot(jacobi_k.T, (spectrum - target_spectrum))
        # 更新折射率，吸收率的系数
        start = time.perf_counter()
        change_n_count = 0
        for i in range(layer_number):
            if n_indices[i, 0] < 5 and partial_n[i, 0] < -0.1:
                n_indices[i, 0] += 1
                change_n_count += 1
            elif n_indices[i, 0] > 1 and partial_n[i, 0] > 0.1:
                n_indices[i, 0] -= 1
                change_n_count += 1
            # if n_indices[i, 1] != 0:
            #     # 只有Nb2O5有吸收
            #     if n_indices[i, 1] < 3 and partial_k[i, 0] < -0.01:
            #         n_indices[i, 1] += 1
            #     elif n_indices[i, 1] > 1 and partial_k[i, 0] > -0.01:
            #         n_indices[i, 1] -= 1
        print('partial_n:')
        print(partial_n)
        end = time.perf_counter()
        print(f'changing the refractive index takes {end-start}, {change_n_count} layers changed')
        # 因为改了折射率，要更新一下梯度下降的参数
        # 用求导算jacobi, 先算一个初始值
        jacobi_d = get_jacobi(wls, d, n_indices)[0:, 0:layer_number]
        # spectrum 是 2N by 1 向量
        spectrum = get_spectrum(wls, d, n_indices)

        f = spectrum - target_spectrum
        g = dot(jacobi_d.T, f)
        A = dot(jacobi_d.T, jacobi_d)
        mu = 1
        nu = 2

        # 存到文件
        structure = pd.DataFrame([['(name and affiliation)', ''], ['email address', ''], ['', '']], columns=[1, 2])
        for i in range(n_indices.shape[0]):
            if n_indices[layer_number - 1 - i, 1] == 0:
                # SiO2 Ln(i),d(i)
                n = n_indices[layer_number - 1 - i, 0]
                structure = pd.concat(
                    [structure, pd.DataFrame({1: [f'L{n}'], 2: [f'{d[layer_number - 1 - i, 0].real}']})])
            else:
                # Nb2O5 Hn(i)k(i), d(i)
                n = int(n_indices[layer_number - 1 - i, 0])
                k = int(n_indices[layer_number - 1 - i, 1])
                structure = pd.concat(
                    [structure, pd.DataFrame({1: [f'H{n}{k}'], 2: [f'{d[layer_number - 1 - i, 0].real}']})])
        structure.to_csv(
            f'C:\\Users\\weenming\\OneDrive\\Ellipsometry\\OIC Design Challenge\\A1\\result\\layernumber_{layer_number}_error_{error}_n.txt',
            index=False, header=None, sep=' ')

if __name__ == '__main__':
    main_gradient()
