import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
sys.path.append("./")
import os
import time
import film as film
import spectrum
import utils.get_n as get_n
import tmm.get_jacobi_adjoint as get_jacobi
from tmm.get_jacobi_n_adjoint import get_jacobi_free_form
import tmm.tmm_cpu.get_jacobi as get_jacobi_cpu
import timeit
from optimizer.grad_helper import stack_J, stack_init_params
import matplotlib.pyplot as plt


def jacobi_adjoint_GPU(layer_number):
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 128000)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, layer_number))
    get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(),
                                 f.calculate_n_array(wls), f.spectra[0].n_sub, f.spectra[0].n_inc, inc_ang, jacobi.shape[1])


def jacobi_n_adjoint_GPU(layer_number):
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 128000)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, layer_number))
    get_jacobi_free_form(jacobi, wls, f.get_d(),
                         f.calculate_n_array(wls), f.spectra[0].n_sub, f.spectra[0].n_inc, inc_ang)


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
        f"jacobi_adjoint_GPU({layer_number})", number=N, setup="from __main__ import jacobi_adjoint_GPU")
    print(f"GPU, spectrum 500 wls: {t_gpu / N}")

    t_cpu = timeit.timeit(
        f"jacobi_CPU({layer_number})", number=N, setup="from __main__ import jacobi_CPU")
    print(f"CPU, spectrum 500 wls: {t_cpu / N}")

    return t_gpu / N, t_cpu / N


def dif_n_GPU(layer_number):
    N = 1
    t_gpu = timeit.timeit(
        f"jacobi_adjoint_GPU({layer_number})", number=N, setup="from __main__ import jacobi_adjoint_GPU")
    print(f"GPU adjoint, spectrum 500 wls: {t_gpu / N}, {layer_number} layers")
    t_n_GPU = timeit.timeit(
        f"jacobi_n_adjoint_GPU({layer_number})", number=N, setup="from __main__ import jacobi_n_adjoint_GPU")
    print(
        f"GPU adjoint w.r.t. n, spectrum 500 wls: {t_n_GPU / N}, {layer_number} layers")

    return t_gpu / N, t_n_GPU / N

if __name__ == "__main__":
    # warm up
    jacobi_adjoint_GPU(100)
    for _ in range(1000):
        dif_n_GPU(100)

