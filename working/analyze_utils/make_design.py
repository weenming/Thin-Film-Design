import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./../../designer/script/')
sys.path.append('./../')
sys.path.append('./../../')
from spectrum import Spectrum
from design import BaseDesign
from film import TwoMaterialFilm


whatever_film = TwoMaterialFilm('1', '2', '1', np.array([1, 1]))


def make_edgefilter_design(
        init_film=whatever_film,
        wls=np.linspace(400, 1000, 500),
        inc_angs=[0]
):
    specs = []
    for inc in inc_angs:
        specs.append(get_edge_filter_design(inc, wls))
    design = BaseDesign(specs, init_film)
    return design


def make_reflection_design(
        init_film=whatever_film,
        wls=np.linspace(695, 939, 500)
) -> BaseDesign:

    design = BaseDesign([get_reflector_spec(wls=wls)], init_film)
    return design


def make_triband_filter_design(init_film=whatever_film):
    design = BaseDesign([get_triband_filter_spec()], init_film)
    return design


def get_minus_filter_spec(wls):
    '''Settings adpted from Jinlong Zhang et al. Thin-film thickness-modulated designs for optical minus filter, 2013
    '''
    assert np.min(wls) < 510 and np.max(wls) > 555, 'wls must cover stop band'
    h = 0.8
    R = (wls > 510.) & (555. > wls)
    R = R.astype(float)
    R *= h
    T = 1 - R
    return Spectrum(0., wls, R, (T + 1e-5))


def get_reflector_spec(inc=0., wls=np.linspace(695, 939, 500)):
    inc_ang = inc
    # wls = np.linspace(700, 800, 500) # when wls = 50, ~100 min
    # default: DBR for SiO2 / TiO2
    R = np.ones(wls.shape[0], dtype='float')
    return Spectrum(inc_ang, wls, R)


def get_edge_filter_design(inc, wls):
    # R = np.ones(wls.shape[0] , dtype='float')
    R = np.zeros(wls.shape[0], dtype='float')
    R[wls.shape[0] // 2:] = 1.

    return Spectrum(inc, wls, R)


def get_triband_filter_spec():
    inc_ang = 0.

    def make_r_spec(wl_1, wl_2):
        wls = np.linspace(wl_1, wl_2, 10 * int(wl_2 - wl_1))
        R = np.ones(wls.shape[0], dtype='float')
        return R

    def make_t_spec(wl_1, wl_2):
        wls = np.linspace(wl_1, wl_2, 10 * int(wl_2 - wl_1))
        T = np.zeros(wls.shape[0], dtype='float')
        return T

    def make_wl(x1, x2): return np.linspace(
        x1, x2, 10 * int(x2 - x1), dtype='float')

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

    return Spectrum(inc_ang, wls, R)
