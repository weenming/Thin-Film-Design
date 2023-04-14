import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
sys.path.append("./")


import os
import copy
import film as film
import gets.get_n as get_n
import gets.get_jacobi as get_jacobi

import matplotlib.pyplot as plt


class TestJacobi(unittest.TestCase):
    def test_film_jacobi(self):
        np.random.seed(1)
        d_expected = np.random.random(30) * 100

        substrate = A = "SiO2"
        B = "TiO2"
        f = film.TwoMaterialFilm(A, B, substrate, d_expected)
        # must set spec before calculating spec
        inc_ang = 60.  # incident angle in degree
        wls = np.linspace(500, 1000, 500)
        f.add_spec_param(inc_ang, wls)

        jacobi = np.empty((wls.shape[0] * 2, 30))
        get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(),
                                     f.spectrums[0].n, f.spectrums[0].n_sub, f.spectrums[0].n_inc, inc_ang, jacobi.shape[1])

        # read expected spec from file
        # count relative path from VS Code project root.....
        expected_jacobi = np.loadtxt(
            "./designer/tests/test_files/expected_jacobi_simple_R_500to1000_30layer_SiO2-TiO2-times-15-SiO2_60inc.csv", dtype="float")
        np.testing.assert_almost_equal(
            jacobi[:wls.shape[0], :], -jacobi[wls.shape[0]:, :])
        np.testing.assert_almost_equal(
            jacobi[:wls.shape[0], :], expected_jacobi[:wls.shape[0], :] / 2)
        np.testing.assert_almost_equal(
            jacobi[wls.shape[0]:, :], expected_jacobi[wls.shape[0]:, :] / 2)

    def test_film_jacobi_many_layers(self):
        np.random.seed(1)
        d_expected = np.random.random(1000) * 10

        substrate = A = "SiO2"
        B = "TiO2"
        f = film.TwoMaterialFilm(A, B, substrate, d_expected)
        # must set spec before calculating spec
        inc_ang = 60.  # incident angle in degree
        wls = np.linspace(500, 1000, 500)
        f.add_spec_param(inc_ang, wls)

        jacobi = np.empty((wls.shape[0] * 2, 1000))
        stack_J(jacobi, [[f.spectrums[0].n, f.spectrums[0].n_sub,
                f.spectrums[0].n_inc]], f.get_d(), f.get_all_spec_list())

        # read expected spec from file
        # count relative path from VS Code project root.....
        expected_jacobi = np.loadtxt(
            "./designer/tests/test_files/expected_jacobi_simple_R_500to1000_1000layer_SiO2-TiO2-times-500-SiO2_60inc.csv", dtype="float")

        # np.testing.assert_almost_equal(jacobi[:wls.shape[0], :], -jacobi[wls.shape[0]:, :])
        np.testing.assert_almost_equal(
            jacobi[:wls.shape[0], :], expected_jacobi[:wls.shape[0], :] / 2)
        np.testing.assert_almost_equal(
            jacobi[wls.shape[0]:, :], expected_jacobi[wls.shape[0]:, :] / 2)


def test_film_jacobi_debug():
    np.random.seed(1)
    d_expected = np.random.random(30) * 100

    substrate = A = "SiO2"
    B = "TiO2"
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 500)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, 30))
    get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(),
                                 f.spectrums[0].n, f.spectrums[0].n_sub, f.spectrums[0].n_inc, inc_ang, jacobi.shape[1])

    print(jacobi)


def make_var_jacobi():
    np.random.seed(1)
    layers = 1000
    d_expected = np.random.random(layers) * 10

    substrate = A = "SiO2"
    B = "TiO2"
    f = film.TwoMaterialFilm(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    wls = np.linspace(500, 1000, 500)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, layers))
    R_1 = np.empty(wls.shape[0] * 2)
    R_2 = np.empty(wls.shape[0] * 2)
    h = 1e-5  # must be neither too big nor too small
    for j in range(jacobi.shape[1]):
        d = copy.deepcopy(f.get_d())
        get_spectrum_simple(R_1, wls, d, f.get_spec().n,
                            f.get_spec().n_sub, f.get_spec().n_inc, inc_ang)
        d[j] += h
        get_spectrum_simple(R_2, wls, d, f.get_spec().n,
                            f.get_spec().n_sub, f.get_spec().n_inc, inc_ang)

        jacobi[:, j] = (R_2 - R_1) / h
    np.savetxt("./designer/tests/test_files/expected_jacobi_simple_R_500to1000_1000layer_SiO2-TiO2-times-500-SiO2_60inc.csv", jacobi)


if __name__ == "__main__":
    make_var_jacobi()
    unittest.main()
