import sys
sys.path.append('./designer/script/')

import unittest
import numpy as np
import matplotlib.pyplot as plt

from film import FilmSimple
from spectrum import Spectrum
from optimizer.LM_gradient_descent import LM_optimize_d_simple
from design import DesignForSpecSimple



def make_design():
    wls = np.linspace(400, 1000, 1000)
    target_spec_R = np.ones(wls.shape[0], dtype='float')
    target_spec = Spectrum(0., wls, target_spec_R)

    d_init = np.random.random(100) * 100
    init_film = FilmSimple('SiO2', 'TiO2', 'SiO2', d_init)

    design = DesignForSpecSimple(target_spec, init_film)
    return design

def LM_descent(design: DesignForSpecSimple):
    LM_optimize_d_simple(design.film, design.target_specs, 1e-200, 1000)



if __name__ == '__main__':
    for seed in [0, 100, 233, 42, 555] + list(np.arange(20)):
        print(f'seed: {seed}')
        np.random.seed(seed)
        design = make_design()
        LM_descent(design)