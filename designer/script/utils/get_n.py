from numpy import *
import numpy as np
from numpy.typing import NDArray
import scipy


import importlib.resources as pkg_resources  # python > 3.7
from designer import material_data


# use these wrapper functions to select which model / exp data to use
def get_n_SiO2(wl):
    return get_n_SiO2_Cauchy(wl)


def get_n_TiO2(wl):
    return get_n_TiO2_Cauchy(wl)


def get_n_Si(wl):
    return get_Si_exp(wl)


def get_n_1(wl):
    return wl / wl  # broadcast if is instance of np.array


def get_n_1_5(wl):
    return 1.5 * wl / wl



def get_n_2(wl):
    return 2 * wl / wl


def get_n_free(wl, n: complex):
    return wl / wl * n


def get_n_SiO2_Cauchy(wl):
    wl = wl * 1e-3  # nm to \mu m
    # SiO2: Ghosh 1999 crystal, alpha-quartz
    # NOTE: applicable to 198 nm - 2050 nm
    return np.sqrt(1.28604141 + 1.07044083 * wl**2 /
                   (wl**2 - 0.0100585997) + 1.10202242 * wl**2 / (wl**2 - 100.))


def get_n_TiO2_Cauchy(wl):
    wl = wl * 1e-3
    # TiO2: Devore 1951, crystal
    # NOTE: applicable to 430 nm - 1530 nm
    return np.sqrt(5.913 + 0.2441 / (wl**2 - 0.0803))


def get_n_Air(wl):
    # approximate
    return 1.


def load_from_file(fname_n, fname_k) -> tuple[NDArray, NDArray]:
    try:
        wls, n = np.loadtxt(
            pkg_resources.read_text(material_data, fname_n).split(),
            dtype='float, float',
            skiprows=1,
            unpack=True,
            delimiter=','
        )
        wls_2, k = np.loadtxt(
            pkg_resources.read_text(material_data, fname_k).split(),
            dtype='float, float',
            skiprows=1,
            unpack=True,
            delimiter=','
        )
    except Exception as e:
        print(e)
        raise ValueError("bad file")
    assert np.array_equal(wls, wls_2), 'wls must be the same'
    # note: return wl in nm
    return wls * 1000, n - 1j * k


cached_Si = False
wls_Si = None
n_Si = None
n_Si_interp = None


def get_Si_exp(wl):
    '''
    Get refractive index & extinction coeff from file.
    NOTE: input wls instead of iteratively calling this function
    has significantly better performance (for ~ 1000 pts, ~1 s 
    compared to ~ 0.003 s)

    Parameters:
        wl: wavelength OR wavelengths (array-like) to compute wl
    '''
    # Si: Green 2008
    # NOTE: 300 nm - 1510 nm

    global cached_Si, wls_Si, n_Si
    if not cached_Si:
        wls_Si, n_Si = load_from_file(
            'Si_n_Green-2008.csv',
            'Si_k_Green-2008.csv'
        )
        n_Si_interp = scipy.interpolate.interp1d(wls_Si, n_Si)

    return n_Si_interp(wl)
