import sys
import timeit

sys.path.append('./')
sys.path.append('./designer/script/')


from tmm.get_E import get_E_free
from tmm.get_jacobi_arb_adjoint import get_jacobi_E_free_form
from tmm.get_jacobi_adjoint import get_jacobi_adjoint
from tmm.get_jacobi_n_adjoint import get_jacobi_free_form
from tmm.get_jacobi import get_jacobi_simple
import tmm.tmm_cpu.get_jacobi as get_jacobi_cpu
from tmm.get_spectrum import get_spectrum_free
from film import FreeFormFilm, TwoMaterialFilm
from tmm.autograd_wrapper import *
from tmm.E_to_spectrum import *

from optimizer.adam import AdamThicknessOptimizerAutograd

import matplotlib.pyplot as plt
import numpy as np
import copy


def jacobi_TFNN_GPU(layer_number):
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 100)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, layer_number))
    get_jacobi_simple(
        jacobi, 
        wls, 
        f.get_d(),
        f.spectra[0].film.calculate_n_array(wls), 
        f.spectra[0].n_sub, 
        f.spectra[0].n_inc, 
        inc_ang, 
        jacobi.shape[1]
    )


def jacobi_TFNN_CPU(layer_number):
    assert layer_number % 2 == 0
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 100)
    f.add_spec_param(inc_ang, wls)

    materials = np.array([A, B] * (layer_number // 2))
    get_jacobi_cpu.get_jacobi(wls, f.get_d(), materials, theta0=inc_ang)

def jacobi_adjoint_GPU(layer_number):
    np.random.seed(1)
    d_expected = np.random.random(layer_number) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 100)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, layer_number))
    # get_jacobi_free_form(
    #     jacobi, 
    #     wls, 
    #     f.get_d(),
    #     f.spectra[0].film.calculate_n_array(wls), 
    #     f.spectra[0].n_sub, 
    #     f.spectra[0].n_inc, 
    #     inc_ang, 
    #     jacobi.shape[1]
    # )
    get_jacobi_adjoint(
        jacobi, 
        wls, 
        f.get_d(),
        f.spectra[0].film.calculate_n_array(wls), 
        f.spectra[0].n_sub, 
        f.spectra[0].n_inc, 
        inc_ang, 
        jacobi.shape[1]
    )


def timer(n_ls):
    adjoint_GPU = []
    TFNN_GPU = []
    TFNN = []

    jacobi_TFNN_GPU(100)
    jacobi_adjoint_GPU(100)

    for layer_number in n_ls:
        t_adjoint_gpu = timeit.timeit(
        f"jacobi_adjoint_GPU({layer_number})", number=10, setup="from __main__ import jacobi_adjoint_GPU")
        print(f"Adjoint GPU, spectrum 500 wls: {t_adjoint_gpu / 10}")
        adjoint_GPU.append(t_adjoint_gpu / 20)

        if layer_number <= 1000:
            t_tfnn_cpu = timeit.timeit(
            f"jacobi_TFNN_CPU({layer_number})", number=10, setup="from __main__ import jacobi_TFNN_CPU")
            print(f"TFNN CPU, spectrum 500 wls: {t_tfnn_cpu / 1}")
            TFNN.append(t_tfnn_cpu)
        
        if layer_number <= 100:
            t_tfnn_gpu = timeit.timeit(
            f"jacobi_TFNN_GPU({layer_number})", number=10, setup="from __main__ import jacobi_TFNN_GPU")
            print(f"TFNN GPU, spectrum 500 wls: {t_tfnn_gpu / 10}")
            TFNN_GPU.append(t_tfnn_gpu / 20)



    np.save('time_result_adjoint_GPU', np.array(adjoint_GPU))
    np.save('time_result_TFNN_GPU', np.array(TFNN_GPU))
    np.save('time_result_TFNN_CPU', np.array(TFNN))
    np.save('time_result_layernumber', np.array(n_ls))
    
if __name__ == '__main__':
    timer([2, 6, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])