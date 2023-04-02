import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
import os
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
        f = film.FilmSimple(A, B, substrate, d_expected)
        # must set spec before calculating spec
        inc_ang = 60. # incident angle in degree
        wls = np.linspace(500, 1000, 500)
        f.add_spec_param(inc_ang, wls)

        jacobi = np.empty((wls.shape[0] * 2, 30))
        get_jacobi.get_jacobi_simple(jacobi, wls, f.get_d(), \
            f.spectrum[0].n, f.spectrum[0].n_sub, f.spectrum[0].n_inc, inc_ang)


        # read expected spec from file
        # count relative path from VS Code project root.....
        expected_jacobi = np.loadtxt("./designer/tests/test_files/expected_jacobi_simple_R_500to1000_30layer_SiO2-TiO2-times-15-SiO2_60inc.csv", dtype="float")
        self.assertAlmostEqual(np.max(np.abs(jacobi[:wls.shape[0], :] + jacobi[wls.shape[0]:, :])), 0)
        self.assertAlmostEqual(np.max(np.abs(jacobi[:wls.shape[0], :] - expected_jacobi / 2)), 0)
        self.assertAlmostEqual(np.max(np.abs(-jacobi[wls.shape[0]:, :] - expected_jacobi / 2)), 0)
        

def test_film_jacobi_debug():
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
        f.spectrum[0].n, f.spectrum[0].n_sub, f.spectrum[0].n_inc, inc_ang)
    
    print(jacobi)

if __name__ == "__main__":
    unittest.main()