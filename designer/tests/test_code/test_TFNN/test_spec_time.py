import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
import timeit
import film as film
import utils.get_n as get_n
import tmm.tmm_cpu.get_spectrum as get_spectrum_cpu

import matplotlib.pyplot as plt


def spec_gpu(layer_number):
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100
    
    substrate = A = "SiO2"
    B = "TiO2"
    f = film.FilmSimple(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60. # incident angle in degree
    wls = np.linspace(500, 1000, 500)

    # calculate
    f.add_spec_param(inc_ang, wls)
    f.calculate_spectrum()


def spec_cpu(layer_number):
    
    # layer number must be even
    assert layer_number % 2 == 0
    
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100
    
    substrate = A = "SiO2"
    B = "TiO2"
    f = film.FilmSimple(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60. # incident angle in degree
    wls = np.linspace(500, 1000, 500)

    materials = np.array([A, B] * (layer_number // 2))
    # calculate
    get_spectrum_cpu.get_spectrum(wls, d_expected, materials, theta0=inc_ang)

def dif_n(layer_number):
    N = 1
    t_gpu = timeit.timeit(f"spec_gpu({layer_number})", number=N, setup="from __main__ import spec_gpu")
    print(f"GPU, spectrum 500 wls: {t_gpu / N}")
    
    t_cpu = timeit.timeit(f"spec_cpu({layer_number})", number=N, setup="from __main__ import spec_cpu")
    print(f"CPU, spectrum 500 wls: {t_cpu / N}")

    return t_gpu / N, t_cpu / N

def plot_time():
    # compile first
    dif_n(10)

    # start testing at different layer numbers
    Ns = np.array(range(10, 100, 20))
    print(Ns)
    Ts_GPU = []
    Ts_CPU = []
    for i in Ns:
        g, c = dif_n(i)
        Ts_CPU.append(c)
        Ts_GPU.append(g)
    
    k_cpu, b_cpu = np.polyfit(Ns, Ts_CPU, 1)
    k_gpu, b_gpu = np.polyfit(Ns, Ts_GPU, 1)
    N_e = np.linspace(0, Ns[-1], 100)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(Ns, Ts_CPU, label='CPU', marker='x')
    ax.plot(N_e, k_cpu * N_e + b_cpu, label='CPU fit')

    ax.scatter(Ns, Ts_GPU, label='GPU', marker='x')
    ax.plot(N_e, k_gpu * N_e + b_gpu, label='GPU fit')

    ax.legend()
    ax.set_xlabel("layer number")
    ax.set_ylabel("time / s")
    ax.set_yscale('log')
    ax.set_xlim(0, Ns[-1])
    plt.show()

if __name__ == "__main__":
    dif_n(2)
    dif_n(200)
    # keep gpu occupied?
    t = timeit.timeit(f"spec_gpu({200})", number=100, setup="from __main__ import spec_gpu")
    print(t)