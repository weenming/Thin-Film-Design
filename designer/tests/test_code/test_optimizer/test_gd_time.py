import sys
sys.path.append('./designer/script/')
sys.path.append('./')

from design import DesignSimple, DesignForSpecSimple
from optimizer.arxiv.adam_d import adam_optimize
from optimizer.adam_non_sgd import adam_optimize_non_sgd
from optimizer.LM_gradient_descent import LM_optimize_d_simple
from spectrum import Spectrum
from film import TwoMaterialFilm
import matplotlib.pyplot as plt
import numpy as np
import time
import unittest


def make_design():
    wls = np.linspace(400, 1000, 1000, dtype='float')
    target_spec_R = np.ones(wls.shape[0], dtype='float')
    target_spec = Spectrum(0., wls, target_spec_R)

    d_init = np.random.random(50) * 100.
    init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)

    design = DesignForSpecSimple(target_spec, init_film)
    return design


def make_design_multi_spec():
    wls = np.linspace(400, 1000, 1000, dtype='float')
    target_spec_R = np.ones(wls.shape[0], dtype='float')
    target_spec1 = Spectrum(0., wls, target_spec_R)
    target_spec2 = Spectrum(45., wls, target_spec_R)
    target_spec3 = Spectrum(60., wls, target_spec_R)

    d_init = np.random.random(50) * 100.
    init_film = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d_init)

    design = DesignSimple(
        [target_spec1, target_spec2, target_spec3], init_film)
    return design


N = 100


def time_gd(gd):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        gd(*args, **kwargs)
        t2 = time.time()
        print(f'takes {(t2 - t1) / N} s')
    return wrapper


@time_gd
def Adam_descent_test(
    design: DesignForSpecSimple,
    wl_batch,
    spec_batch
):

    adam_optimize(
        design.film,
        design.target_specs,
        max_steps=N,
        alpha=1,
        show=False,
        batch_size_spec=spec_batch,
        batch_size_wl=wl_batch
    )
    '''
    As layer number increases, step size needs to decrease
    '''


@time_gd
def Non_SGD_Adam_descent_test(
    design: DesignForSpecSimple,
):
    adam_optimize_non_sgd(design.film, design.target_specs,
                          max_steps=N, alpha=1, show=False)

    '''
    As layer number increases, step size needs to decrease
    '''


if __name__ == '__main__':
    print('warm up')
    Non_SGD_Adam_descent_test(make_design())
    Adam_descent_test(make_design(), None, None)

    print('end of warm up\n')

    print('Single spectrum:\nSGD, no minibatch')
    Adam_descent_test(make_design(), None, None)
    print('SGD, minibatch 10%')
    Adam_descent_test(make_design(), 200, None)
    print('SGD, minibatch 50%')
    Adam_descent_test(make_design(), 500, None)
    print('SGD, minibatch 100%')
    Adam_descent_test(make_design(), 1000, None)
    print('non SGD:')
    Non_SGD_Adam_descent_test(make_design())
    # Result: no significant decrease by using SGD.
    # 50 layers @ 1000 wls, ~0.01s per epoch
    # 200 layers @ 1000 wls, ~0.03s per epoch
    # 500 layers @ 1000 wls, ~0.08s per epoch

    print('3 Spectrums:\nSGD, no minibatch')
    Adam_descent_test(make_design_multi_spec(), None, None)
    print('SGD, minibatch 100% / 33%')
    Adam_descent_test(make_design_multi_spec(), 1000, 1)
    print('SGD, minibatch 100% / 66%')
    Adam_descent_test(make_design_multi_spec(), 1000, 2)
    print('SGD, minibatch 100% / 100%')
    Adam_descent_test(make_design_multi_spec(), 1000, 3)

    print('non SGD:')
    Non_SGD_Adam_descent_test(make_design_multi_spec())
    # Result: SGD saves time when calculating multi spec,
    # because parallelization of specs is not yet implemented
