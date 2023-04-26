import sys
sys.path.append('./designer/script/')
sys.path.append('./')

from design import DesignForSpecSimple
from optimizer.arxiv.adam_d import adam_optimize
from optimizer.arxiv.adam_non_sgd import adam_optimize_non_sgd
from optimizer.LM_gradient_descent import LM_optimize_d_simple
from optimizer.adam import AdamThicknessOptimizer
from spectrum import Spectrum
from film import TwoMaterialFilm
import matplotlib.pyplot as plt
import numpy as np
import unittest
import copy


class TestGD(unittest.TestCase):
    def test_adam(self):

        wls = np.linspace(400, 1000, 1000, dtype='float')
        target_spec_R = np.ones(wls.shape[0], dtype='float')
        target_spec = Spectrum(0., wls, target_spec_R)

        d_init = np.random.random(500) * 10.  # integer times of MAX
        init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)
        design = DesignForSpecSimple(target_spec, init_film)
        adam_optimize(design.film, design.
                      target_specs, max_steps=50, alpha=0.1, show=True)

        another_film = copy.deepcopy(design.film)
        adam_optim = AdamThicknessOptimizer(another_film, design.
                                            target_specs, max_steps=50, alpha=0.1, show=True)
        adam_optim()
        np.testing.assert_almost_equal(
            design.film.get_d(), another_film.get_d())

    def test_adam_sgd(self):

        wls = np.linspace(400, 1000, 1000, dtype='float')
        target_spec_R = np.ones(wls.shape[0], dtype='float')
        target_spec = Spectrum(0., wls, target_spec_R)

        d_init = np.random.random(500) * 50.  # integer times of MAX
        init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)
        design = DesignForSpecSimple(target_spec, init_film)
        adam_optimize(design.film, design.
                      target_specs, max_steps=50, alpha=0.1, show=True, batch_size_wl=1000)

        d_init = np.random.random(523) * 50.  # not integer times of MAX
        init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)
        design = DesignForSpecSimple(target_spec, init_film)
        adam_optimize(design.film, design.target_specs,
                      max_steps=50, alpha=0.1, show=True, batch_size_wl=501)

    def test_LM(self):

        wls = np.linspace(400, 1000, 1000, dtype='float')
        target_spec_R = np.ones(wls.shape[0], dtype='float')
        target_spec = Spectrum(0., wls, target_spec_R)

        d_init = np.random.random(500) * 50.  # integer times of MAX
        init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)
        design = DesignForSpecSimple(target_spec, init_film)
        LM_optimize_d_simple(
            design.film, design.target_specs, 1e-200, 1000, show=True)

        d_init = np.random.random(523) * 50.  # not integer times of MAX
        init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)
        design = DesignForSpecSimple(target_spec, init_film)
        LM_optimize_d_simple(
            design.film, design.target_specs, 1e-200, 1000, show=True)


def make_design():
    wls = np.linspace(400, 1000, 1000, dtype='float')
    target_spec_R = np.ones(wls.shape[0], dtype='float')
    target_spec = Spectrum(0., wls, target_spec_R)

    d_init = np.random.random(52) * 100.
    init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)

    design = DesignForSpecSimple([target_spec], init_film)
    return design


def LM_descent_test(design: DesignForSpecSimple):
    LM_optimize_d_simple(design.film, design.target_specs,
                         1e-200, 1000, show=True)


def Adam_descent_test(design: DesignForSpecSimple):
    adam_optimize(design.film, design.target_specs,
                  max_steps=5000, alpha=0.01, show=True)
    '''
    As layer number increases, step size needs to decrease
    '''


def Non_SGD_Adam_descent_test(design: DesignForSpecSimple):
    adam_optimize_non_sgd(design.film, design.target_specs,
                          max_steps=50, alpha=1, show=True)
    print(design.film.get_d())
    '''
    As layer number increases, step size needs to decrease
    '''


if __name__ == '__main__':
    for seed in [0, 100, 233, 42, 555] + list(np.arange(20)):
        print(f'seed: {seed}')
        np.random.seed(seed)
        design = make_design()
        Non_SGD_Adam_descent_test(design)
