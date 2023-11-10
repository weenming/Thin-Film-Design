import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
sys.path.append("./")


import os
import copy
import film as film
from tmm.get_jacobi_adjoint import get_jacobi_adjoint
from tmm.get_spectrum import get_spectrum_free, get_spectrum_simple

import matplotlib.pyplot as plt


class TestJacobi(unittest.TestCase):
    def test_film_jacobi(self):
        np.random.seed(1)
        d_expected = np.random.random(1000) * 10

        A = "SiO2"
        B = "TiO2"
        substrate = 2
        f = film.TwoMaterialFilm(A, B, substrate, d_expected)
        # must set spec before calculating spec
        inc_ang = 60.  # incident angle in degree
        wls = np.linspace(500, 1000, 500)
        f.add_spec_param(inc_ang, wls)

        jacobi = np.empty((wls.shape[0] * 2, 1000))
        get_jacobi_adjoint(jacobi, wls, f.get_d(),
                                     f.calculate_n_array(wls), f.calculate_n_sub(wls), f.calculate_n_inc(wls), inc_ang)

        # read expected spec from file
        # count relative path from VS Code project root.....
        expected_jacobi = np.loadtxt(
            "./designer/tests/test_files/expected_jacobi_simple_R_500to1000_1000layer_SiO2-TiO2-times-500-SiO2_60inc.csv", dtype="float")
        np.testing.assert_almost_equal(
            jacobi[:wls.shape[0], :], -jacobi[wls.shape[0]:, :])
        np.testing.assert_almost_equal(
            jacobi[:wls.shape[0], :], expected_jacobi[:wls.shape[0], :] / 2)
        np.testing.assert_almost_equal(
            jacobi[wls.shape[0]:, :], expected_jacobi[wls.shape[0]:, :] / 2)
    
    def test_one_layer(self):
        return

    def test_inc_not1(self):
        return

def make_diff_jacobi():
    np.random.seed(1)
    layers = 1000
    d_expected = np.random.random(layers) * 10

    A = "SiO2"
    B = "TiO2"
    substrate = 2
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
        get_spectrum_simple(R_1, wls, d, f.calculate_n_array(wls),
                            f.calculate_n_sub(wls), f.calculate_n_inc(wls), inc_ang)
        d[j] += h
        get_spectrum_simple(R_2, wls, d, f.calculate_n_array(wls),
                            f.calculate_n_sub(wls), f.calculate_n_inc(wls), inc_ang)

        jacobi[:, j] = (R_2 - R_1) / h
    np.savetxt("./designer/tests/test_files/expected_jacobi_simple_R_500to1000_1000layer_SiO2-TiO2-times-500-SiO2_60inc.csv", jacobi)


if __name__ == "__main__":
    make_diff_jacobi()
    unittest.main()
