import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
import os
import time
import film as film
import spectrum
import utils.get_n as get_n
import tmm.get_jacobi as get_jacobi
import tmm.tmm_cpu.get_jacobi as get_jacobi_cpu
import timeit
from optimizer.grad_helper import stack_J, stack_init_params
import matplotlib.pyplot as plt


def jacobi_GPU(layer_number):
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 500)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, layer_number))
    get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(),
                                 f.spectrums[0].n, f.spectrums[0].n_sub, f.spectrums[0].n_inc, inc_ang, jacobi.shape[1])


def jacobi_CPU(layer_number):
    assert layer_number % 2 == 0
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 500)
    f.add_spec_param(inc_ang, wls)

    materials = np.array([A, B] * (layer_number // 2))
    get_jacobi_cpu.get_jacobi(wls, f.get_d(), materials, theta0=inc_ang)


def dif_n(layer_number):
    N = 1
    t_gpu = timeit.timeit(
        f"jacobi_GPU({layer_number})", number=N, setup="from __main__ import jacobi_GPU")
    print(f"GPU, spectrum 500 wls: {t_gpu / N}")

    t_cpu = timeit.timeit(
        f"jacobi_CPU({layer_number})", number=N, setup="from __main__ import jacobi_CPU")
    print(f"CPU, spectrum 500 wls: {t_cpu / N}")

    return t_gpu / N, t_cpu / N


def dif_n_GPU(layer_number):
    N = 1
    t_gpu = timeit.timeit(
        f"jacobi_GPU({layer_number})", number=N, setup="from __main__ import jacobi_GPU")
    print(f"GPU, spectrum 500 wls: {t_gpu / N}")

    return t_gpu / N


def plot_time():
    # compile first
    dif_n(10)

    # start testing at different layer numbers
    # NOTE: max layer number is set in gets.get_jacobi
    Ns = np.array(range(10, 200, 20))
    Ts_GPU = []
    Ts_CPU = []
    for i in Ns:
        g, c = dif_n(i)
        Ts_CPU.append(c)
        Ts_GPU.append(g)

    k_cpu, b_cpu = np.polyfit(Ns, Ts_CPU, 1)
    k_gpu, b_gpu = np.polyfit(Ns, Ts_GPU, 1)
    N_e = np.linspace(0, Ns[-1], 1000)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(Ns, Ts_CPU, label='CPU', marker='x')
    ax.plot(N_e, k_cpu * N_e + b_cpu, label='CPU fit')

    ax.scatter(Ns, Ts_GPU, label='GPU', marker='x')
    ax.plot(N_e, k_gpu * N_e + b_gpu, label='GPU fit')

    ax.legend()
    ax.set_xlabel("layer number")
    ax.set_ylabel("time / s")
    # ax.set_yscale('log')
    ax.set_xlim(0, Ns[-1])
    plt.show()


def plot_time_GPU():
    # compile first
    dif_n(10)

    # start testing at different layer numbers
    # NOTE: max layer number is set in gets.get_jacobi
    Ns = np.array(range(10, 100, 1))
    print(Ns)
    Ts_GPU = []
    for i in Ns:
        g = dif_n_GPU(i)
        Ts_GPU.append(g)

    k_gpu, b_gpu = np.polyfit(Ns, Ts_GPU, 1)
    N_e = np.linspace(0, Ns[-1], 1000)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(Ns, Ts_GPU, label='GPU', marker='x')
    ax.plot(N_e, k_gpu * N_e + b_gpu, label='GPU fit')
    ax.legend()
    ax.set_xlabel("layer number")
    ax.set_ylabel("time / s")
    # ax.set_yscale('log')
    ax.set_xlim(0, Ns[-1])
    ax.set_ylim(0, k_gpu * N_e[-1] + b_gpu)

    plt.show()


def helper_vs_bare():
    # compile first
    dif_n(10)

    np.random.seed(1)
    d_expected = np.random.random(250) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 500)
    f.add_spec_param(inc_ang, wls)

    target_spec_ls = [spectrum.Spectrum(0., wls, np.ones(wls.shape[0]))]
    _, n_arr_ls = stack_init_params(f, target_spec_ls)
    J = np.empty((wls.shape[0] * 2, f.d.shape[0]))

    t1 = time.time()
    for _ in range(10):
        stack_J(J, n_arr_ls, f.d, target_spec_ls)
    t2 = time.time()
    print(f'helper: {(t2 - t1) / 10}')

    t1 = time.time()
    for _ in range(10):
        get_jacobi.get_jacobi_simple(J, wls, f.d,
                                     f.spectrums[0].n, f.spectrums[0].n_sub, f.spectrums[0].n_inc, inc_ang, J.shape[1])
        t2 = time.time()
    print(f'bare: {(t2 - t1) / 10}')


def test_helper():
    # Runtime: about
    # compile first
    print('warm up (compiling)')
    dif_n(10)
    print('end of warm up')

    substrate = A = "SiO2"
    B = "TiO2"
    d_expected = np.array([1, 2, 3.])
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 500)
    f.add_spec_param(inc_ang, wls)

    target_spec_ls = [spectrum.Spectrum(0., wls, np.ones(wls.shape[0]))]

    t = []
    layer_numbers = list(range(2, 1000, 10)) + list(range(1000, 2000, 30))

    for layer_number in layer_numbers:
        print(layer_number)

        np.random.seed(1)
        d_expected = np.random.random(layer_number) * 100
        f.update_d(d_expected)
        _, n_arr_ls = stack_init_params(f, target_spec_ls)
        J = np.empty((wls.shape[0] * 2, f.d.shape[0]))

        t1 = time.time()
        for _ in range(10):
            stack_J(J, n_arr_ls, f.d, target_spec_ls)
        t2 = time.time()

        t.append((t2 - t1) / 10)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(layer_numbers, t, marker='x')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlim(0, 0.2)
    plt.show()

# TODO: optimize grid size w.r.t. wavelength size


if __name__ == "__main__":
    test_helper()
