import numpy as np
import sys
sys.path.append("./designer/script")
import film as film
import utils.get_n as get_n

import matplotlib.pyplot as plt


d_expected = np.random.random(30) * 100

substrate = A = "SiO2"
B = "TiO2"
f = film.FilmSimple(A, B, substrate, d_expected)
# must set spec before calculating spec
inc_ang = 60. # incident angle in degree
wls = np.linspace(500, 1000, 500)
f.add_spec_param(inc_ang, wls)
f.calculate_spectrum()

print(f.spectrum[0].get_R())