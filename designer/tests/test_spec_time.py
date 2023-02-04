import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
import os
import film as film
import gets.get_n as get_n

import matplotlib.pyplot as plt


def spec_gpu():
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
    
    plt.plot(f.spectrum[0].get_R())
    plt.plot()