
import sys

sys.path.append('./../../../../')
sys.path.append('./../../../script/')
sys.path.append('./')
sys.path.append('./designer/script/')

from tmm.get_E import get_E_free
from tmm.get_jacobi_arb_adjoint import get_jacobi_E_free_form
from tmm.get_jacobi_adjoint import get_jacobi_adjoint
from tmm.autograd_wrapper import *
from tmm.E_to_spectrum import E_to_R, E_to_tan2Psi
from tmm.get_spectrum import get_spectrum_free

from film import FreeFormFilm

import numpy as np
import unittest


class TestJacobi(unittest.TestCase):
    def test_film_jacobi_manual_grad_r_sub_1(self):

        wls = np.linspace(500, 1000, 3)
        inc_ang = 0.
        np.random.seed(0)
        film = FreeFormFilm(np.random.random(4) * 1 + 1.5, 1000, 2)
        jacobian = np.zeros((wls.shape[0] * 4, film.get_d().shape[0], 2, 2), dtype='complex128')

        get_jacobi_E_free_form(
            jacobian, wls, film.get_d(), 
            film.calculate_n_array(wls), 
            film.calculate_n_sub(wls), 
            film.calculate_n_inc(wls), 
            inc_ang, 
        )

        E = np.zeros((wls.shape[0] * 2, 2), dtype='complex128')
        get_E_free(E, wls, film.get_d(), 
            film.calculate_n_array(wls), 
            film.calculate_n_sub(wls), 
            film.calculate_n_inc(wls), 
            inc_ang, 
        )


        jacobian_gt = np.zeros((wls.shape[0] * 2, film.get_d().shape[0]))
        get_jacobi_adjoint(
            jacobian_gt, wls, film.get_d(), 
            film.calculate_n_array(wls), 
            film.calculate_n_sub(wls), 
            film.calculate_n_inc(wls), 
            inc_ang, 
        )

        import cmath
        jacobian_d = np.zeros_like(jacobian)

        def calc_partial_d_M(res_mat_s, res_mat_p, n_inc, inc_ang, ni, di, wl):
            cosi = cmath.sqrt(
                1 - ((n_inc / ni) * cmath.sin(inc_ang)) ** 2)
            phi = 2 * cmath.pi * 1j * cosi * ni * di / wl
            coshi = cmath.cosh(phi)
            sinhi = cmath.sinh(phi)

            res_mat_s[0, 0] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl
            res_mat_s[0, 1] = 2 * cmath.pi * 1j * coshi / wl
            res_mat_s[1, 0] = 2 * cmath.pi * 1j * cosi ** 2 * ni ** 2 * coshi / wl
            res_mat_s[1, 1] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl

            res_mat_p[0, 0] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl
            res_mat_p[0, 1] = 2 * cmath.pi * 1j * ni ** 2 * coshi / wl
            res_mat_p[1, 0] = 2 * cmath.pi * 1j * cosi ** 2 * coshi / wl
            res_mat_p[1, 1] = 2 * cmath.pi * 1j * ni * cosi * sinhi / wl


        for i in list(range(wls.shape[0])) + list(range(wls.shape[0] * 2, wls.shape[0] * 3)):
            for j in range(film.get_d().shape[0]):
                calc_partial_d_M(
                    jacobian_d[i, j], 
                    jacobian_d[i + wls.shape[0], j], 
                    film.calculate_n_inc(wls)[i % 3], 
                    inc_ang,
                    film.calculate_n_array(wls)[i % 3, j], 
                    film.get_d()[j], 
                    wls[i % 3]
                )

        jacobian_r = np.zeros_like(jacobian)
        jacobian_r[wls.shape[0] * 2:, :, :, :] = \
            ((E[:, 1] / E[:, 0]).conjugate() * 1 / E[:, 0]).reshape(-1, 1, 1, 1)
        jacobian_r[:wls.shape[0] * 2, :, :, :] = \
            ((E[:, 1] / E[:, 0]).conjugate() * -E[:, 1] / E[:, 0] ** 2).reshape(-1, 1, 1, 1)

        jacobian_final = \
            (jacobian * jacobian_r * jacobian_d).real.sum((-1, -2)).reshape(-1, 3, film.get_d().shape[0]).sum(0)

        np.testing.assert_almost_equal(jacobian_final, 2 * jacobian_gt[:wls.shape[0], :])

    def test_film_jacobi_autograd_r(self):

        # set exp 
        wls = np.linspace(500, 1000, 500)
        inc_ang = 0.
        np.random.seed(0)
        film = FreeFormFilm(np.random.rand(20) * 1 + 1.5, 1000, 2)
        d = film.get_d()
        n_layers, n_sub, n_inc = film.calculate_n_array(wls), film.calculate_n_sub(wls), film.calculate_n_inc(wls)

        # objective: 0.5
        def loss_fn(E):
            e = E_to_R(E)
            dif = e - 0.5
            return dif.square().sum()

        jacobi_auto = get_jacobi_warpper(loss_fn)
        jacobi = np.zeros((wls.shape[0] * 4, d.shape[0], 2, 2))


        jacobi = jacobi_auto(    
            jacobi,
            wls,
            d,
            n_layers,
            n_sub,
            n_inc,
            inc_ang,    
        )

        jacobian_gt = np.zeros((wls.shape[0] * 2, film.get_d().shape[0]))
        get_jacobi_adjoint(
            jacobian_gt, wls, film.get_d(), 
            film.calculate_n_array(wls), 
            film.calculate_n_sub(wls), 
            film.calculate_n_inc(wls), 
            inc_ang, 
        )
        spec_gt = np.zeros((wls.shape[0] * 2))
        get_spectrum_free(
            spec_gt, 
            wls, 
            film.get_d(), 
            film.calculate_n_array(wls), 
            film.calculate_n_sub(wls), 
            film.calculate_n_inc(wls), 
            inc_ang, 
        )
        grad_gt = 8 * jacobian_gt[:wls.shape[0], :].T * (spec_gt[:wls.shape[0]] - 0.5)

        
        

        grad_diff = np.zeros(film.get_layer_number())
        for i in range(film.get_layer_number()):
            d_original = film.get_d().copy()
            d = d_original.copy()
            d[i] += 1e-5
            film.update_d(d)
            spec_gt_var = np.zeros((wls.shape[0] * 2))
            get_spectrum_free(
                spec_gt_var, 
                wls, 
                film.get_d(), 
                film.calculate_n_array(wls), 
                film.calculate_n_sub(wls), 
                film.calculate_n_inc(wls), 
                inc_ang, 
            )
            film.update_d(d_original)
            grad_diff[i] = (np.square(spec_gt_var - 0.5).sum() - np.square(spec_gt - 0.5).sum()) / 1e-5

        # compare diff with analytic grad
        np.testing.assert_almost_equal(grad_gt.sum(-1), grad_diff)
        # compare autograd with analytic grad
        np.testing.assert_almost_equal(grad_gt.sum(-1), 2 * jacobi.cpu().detach())

    def test_film_autograd_ellipsometry_psi(self):
        '''tan Psi: |rp| / |rs|'''
        return

    def test_film_autograd_ellipsometry_delta(self):
        '''e^i\delta = (rp |rs|) / (|rp| rs)'''
        return


if __name__ == '__main__':
    unittest.main()