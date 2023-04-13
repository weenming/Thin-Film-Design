import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
sys.path.append("./") # wtf?
import os
import film as film
import utils.get_n as get_n
import tmm.get_jacobi_adjoint as get_jacobi

import matplotlib.pyplot as plt



class TestJacobi(unittest.TestCase):
    def test_film_jacobi(self):
        np.random.seed(1)
        d_expected = np.random.random(30) * 100
        
        substrate = A = "SiO2"
        B = "TiO2"
        f = film.FilmSimple(A, B, substrate, d_expected)
        # must set spec before calculating spec
        inc_ang = 60. # incident angle in degree
        wls = np.linspace(500, 1000, 500)
        f.add_spec_param(inc_ang, wls)

        jacobi = np.empty((wls.shape[0] * 2, 30))
        get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(), \
            f.spectrum[0].n, f.spectrum[0].n_sub, f.spectrum[0].n_inc, inc_ang, jacobi.shape[1])


        # read expected spec from file
        # count relative path from VS Code project root.....
        expected_jacobi = np.loadtxt("./designer/tests/test_files/expected_jacobi_simple_R_500to1000_30layer_SiO2-TiO2-times-15-SiO2_60inc.csv", dtype="float")
        self.assertAlmostEqual(np.max(np.abs(jacobi[:wls.shape[0], :] + jacobi[wls.shape[0]:, :])), 0)
        self.assertAlmostEqual(np.max(np.abs(jacobi[:wls.shape[0], :] - expected_jacobi / 2)), 0)
        self.assertAlmostEqual(np.max(np.abs(-jacobi[wls.shape[0]:, :] - expected_jacobi / 2)), 0)
        
    def test_many_layer_film_jacobi(self):
        np.random.seed(1)
        layers = 3000
        d_expected = np.random.random(layers) * 100
        
        substrate = A = "SiO2"
        B = "TiO2"
        f = film.FilmSimple(A, B, substrate, d_expected)
        # must set spec before calculating spec
        inc_ang = 60. # incident angle in degree
        wls = np.linspace(500, 1000, 5000)
        f.add_spec_param(inc_ang, wls)

        jacobi = np.empty((wls.shape[0] * 2, layers))
        get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(), \
            f.spectrum[0].n, f.spectrum[0].n_sub, f.spectrum[0].n_inc, inc_ang, jacobi.shape[1])


def test_film_jacobi_debug():
    np.random.seed(1)
    layer = 10
    d_expected = np.random.random(layer) * 10
    
    substrate = A = "SiO2"
    B = "TiO2"
    f = film.FilmSimple(A, B, substrate, d_expected)
    # must set spec before calculating spec
    inc_ang = 60. # incident angle in degree
    wls = np.linspace(500, 1000, 1)
    f.add_spec_param(inc_ang, wls)

    jacobi = np.empty((wls.shape[0] * 2, layer))
    get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(), \
        f.spectrum[0].n, f.spectrum[0].n_sub, f.spectrum[0].n_inc, inc_ang, jacobi.shape[1])
    
    print(jacobi)

if __name__ == "__main__":
    # test_film_jacobi_debug()
    unittest.main()