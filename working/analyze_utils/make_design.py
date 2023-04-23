import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./../../designer/script/')
sys.path.append('./../')
sys.path.append('./../../')
from spectrum import Spectrum
from design import BaseDesign


def make_edgefilter_design(init_film):
    inc_ang = 0.
    wls = np.linspace(400, 1000, 500)  # when wls = 50, ~100 min
    # R = np.ones(wls.shape[0] , dtype='float')
    R = np.zeros(wls.shape[0], dtype='float')
    R[wls.shape[0] // 2:] = 1.
    target_spec = [Spectrum(inc_ang, wls, R)]

    design = BaseDesign(target_spec, init_film)
    return design


def make_reflection_design(
        init_film,
        wls=np.linspace(695, 939, 500)
) -> BaseDesign:
    inc_ang = 0.
    # wls = np.linspace(700, 800, 500) # when wls = 50, ~100 min
    # default: DBR for SiO2 / TiO2
    R = np.ones(wls.shape[0], dtype='float')
    target_spec = [Spectrum(inc_ang, wls, R)]

    design = BaseDesign(target_spec, init_film)
    return design


def make_three_line_filter_design(init_film):
    inc_ang = 0.

    def make_r_spec(wl_1, wl_2):
        wls = np.linspace(wl_1, wl_2, 2 * int(wl_2 - wl_1))
        R = np.ones(wls.shape[0], dtype='float')
        return R

    def make_t_spec(wl_1, wl_2):
        wls = np.linspace(wl_1, wl_2, 2 * int(wl_2 - wl_1))
        R = np.zeros(wls.shape[0], dtype='float')
        return R

    def make_wl(x1, x2): return np.linspace(
        x1, x2, 2 * int(x2 - x1), dtype='float')

    wls, R = np.array([]), np.array([])

    wls = np.append(wls, make_wl(400, 440))
    R = np.append(R, make_r_spec(400, 440))

    wls = np.append(wls, make_wl(445, 455))
    R = np.append(R, make_t_spec(445, 455))

    wls = np.append(wls, make_wl(460, 500))
    R = np.append(R, make_r_spec(460, 500))

    wls = np.append(wls, make_wl(505, 515))
    R = np.append(R, make_t_spec(505, 515))

    wls = np.append(wls, make_wl(520, 630))
    R = np.append(R, make_r_spec(520, 630))

    wls = np.append(wls, make_wl(635, 645))
    R = np.append(R, make_t_spec(635, 645))

    wls = np.append(wls, make_wl(650, 700))
    R = np.append(R, make_r_spec(650, 700))

    target_spec = [Spectrum(inc_ang, wls, R)]

    design = BaseDesign(target_spec, init_film)
    return design
