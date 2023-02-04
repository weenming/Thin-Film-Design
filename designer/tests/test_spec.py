import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
import os
import film as film
import gets.get_n as get_n

import matplotlib.pyplot as plt



class TestFilm(unittest.TestCase):

    def test_film_init_1layer(self):
        d_expected = np.array([100.])
        substrate = A = "SiO2"
        B = "TiO2"
        f = film.FilmSimple(A, B, substrate, d_expected)
        self.assertAlmostEqual(f.get_d(), d_expected)

    def test_film_init_100layers(self):
        d_expected = np.array([100.] * 100)
        substrate = A = "SiO2"
        B = "TiO2"
        f = film.FilmSimple(A, B, substrate, d_expected)
        self.assertAlmostEqual(np.max(np.abs(f.get_d() - d_expected)), 0)
 

    def test_film_change_d(self):
        layer_number = 100
        d_expected = np.array([i + 0.1 for i in range(layer_number * 2)])
        wls = np.linspace(500., 1000., 500)
        inc_ang = 30.
        
        substrate = A = "SiO2"
        B = "TiO2"
        
        n_expected = np.array([[get_n.get_n_SiO2(wl), get_n.get_n_TiO2(wl)] * \
            layer_number for wl in wls])
        f = film.FilmSimple(A, B, substrate, d_expected)
        f.add_spec_param(inc_ang, wls)

        self.assertAlmostEqual(np.max(np.abs(f.get_d() - d_expected)), 0)
        self.assertAlmostEqual(np.max(np.abs(f.calculate_n_array(wls) - n_expected)), 0)

    def test_film_spectrum(self):
        np.random.seed(1)
        d_expected = np.random.random(30) * 100
        
        substrate = A = "SiO2"
        B = "TiO2"
        f = film.FilmSimple(A, B, substrate, d_expected)
        # must set spec before calculating spec
        inc_ang = 60. # incident angle in degree
        wls = np.linspace(500, 1000, 500)
        f.add_spec_param(inc_ang, wls)
        f.calculate_spectrum()

        # read expected spec from file
        # count relative path from VS Code project root.....
        expected_spec = np.loadtxt("./designer/tests/test_files/expected_spectrum_simple_R_500to1000_30layer_SiO2-TiO2-times-15-SiO2_60inc.csv", dtype="float")
        self.assertAlmostEqual(np.max(np.abs(f.spectrum[0].get_R() - expected_spec)), 0)
        self.assertAlmostEqual(np.max(np.abs(1 - f.spectrum[0].get_T() - expected_spec)), 0)

        plt.plot(f.spectrum[0].get_R())
        plt.plot()
    
if __name__ == "__main__":
    unittest.main()