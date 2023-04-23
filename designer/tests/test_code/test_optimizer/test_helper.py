import sys
sys.path.append('./designer/script/')
sys.path.append('./')

from design import DesignForSpecSimple
from spectrum import Spectrum
from film import TwoMaterialFilm

from optimizer.grad_helper import stack_f, stack_J, stack_init_params

from tmm.get_jacobi import get_jacobi_simple
from tmm.get_spectrum import get_spectrum_simple

import matplotlib.pyplot as plt
import numpy as np
import unittest


wls = np.linspace(500, 100, 1)


def make_standard_d():
    # 4 layers
    return np.array([100] * 2)


def make_target():
    target_film = TwoMaterialFilm('1', '2', '1', make_standard_d())

    target_film.add_spec_param(0., wls)
    target_film.add_spec_param(60., wls)
    return target_film


def make_film():
    film = TwoMaterialFilm('1', '2', '1', make_standard_d() + 1)
    return film


def params(film, target_film):
    n_arrs_ls = stack_init_params(film, target_film.get_all_spec_list())
    wl_n = sum([len(x.WLS) * 2 for x in target_film.get_all_spec_list()])
    return n_arrs_ls, wl_n


class TestHelper(unittest.TestCase):

    def test_stack_f_J(self):
        target = make_target()
        film = make_film()
        n_arrs_ls, wl_n = params(film, target)

        f = np.empty(wl_n)
        stack_f(
            f,
            n_arrs_ls,
            film.get_d(),
            target.get_all_spec_list(),
        )

        J = np.empty((wl_n, film.get_d().shape[0]))
        stack_J(
            J,
            n_arrs_ls,
            film.get_d(),
            target.get_all_spec_list(),
        )

        f_manual = np.zeros(wl_n) + 100.
        J_manual = np.zeros((wl_n, film.get_d().shape[0])) + 100.
        i = 0
        for s in target.get_all_spec_list():
            get_spectrum_simple(
                f_manual[i: i + s.WLS.shape[0] * 2],
                s.WLS,
                film.get_d(),
                film.calculate_n_array(s.WLS),
                film.calculate_n_sub(s.WLS),
                film.calculate_n_inc(s.WLS),
                s.INC_ANG
            )
            f_manual[i: i + s.WLS.shape[0]] -= s.get_R()
            f_manual[i + s.WLS.shape[0]: i + s.WLS.shape[0]
                     * 2] -= s.get_T()

            get_jacobi_simple(
                J_manual[i: i + s.WLS.shape[0] * 2],
                s.WLS,
                film.get_d(),
                film.calculate_n_array(s.WLS),
                film.calculate_n_sub(s.WLS),
                film.calculate_n_inc(s.WLS),
                s.INC_ANG,
            )

            i += s.WLS.shape[0] * 2
        np.testing.assert_almost_equal(f, f_manual)
        np.testing.assert_almost_equal(J, J_manual)


if __name__ == '__main__':
    unittest.main()
