import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
sys.path.append("./")
import os
import film as film
import utils.get_n as get_n
import tmm.tmm_cpu.get_spectrum as get_spectrum_cpu
import tmm.get_spectrum as get_spectrum

import matplotlib.pyplot as plt


wls = np.linspace(500, 1000, 500)


class TestFilm(unittest.TestCase):

    def test_film_spectrum(self):

        n = np.array([1., 2.] * 10, dtype='complex128')
        f = film.FreeFormFilm(n, 2000, "SiO2")
        # must set spec before calculating spec
        inc_ang = 60.  # incident angle in degree
        f.add_spec_param(inc_ang, wls)
        f.calculate_spectrum()

        # read expected spec from file
        # count relative path from VS Code project root.....
        expected_spec = np.loadtxt(
            "./designer/tests/test_files/expected_spectrum_simple_R_500to1000_20layer_1-2-times-10-SiO2_60inc.csv", dtype="float")
        np.testing.assert_almost_equal(
            f.spectra[0].get_R(), expected_spec[:wls.shape[0]])
        np.testing.assert_almost_equal(
            f.spectra[0].get_T(), expected_spec[wls.shape[0]:])


def make_expected_spec():
    n = np.array([1., 2.] * 10, dtype='complex128')
    f = film.FreeFormFilm(n, 2000, "SiO2")
    # must set spec before calculating spec
    inc_ang = 60.  # incident angle in degree
    f.add_spec_param(inc_ang, wls)

    spec = np.empty(wls.shape[0] * 2, dtype='float')
    get_spectrum.get_spectrum_simple(spec, wls, f.get_d(), f.calculate_n_array(
        wls), f.calculate_n_sub(wls), f.calculate_n_inc(wls), f.get_spec().INC_ANG)

    np.savetxt(
        './designer/tests/test_files/expected_spectrum_simple_R_500to1000_20layer_1-2-times-10-SiO2_60inc.csv', spec)


if __name__ == "__main__":
    make_expected_spec()
    unittest.main()
