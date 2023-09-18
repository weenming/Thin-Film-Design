from numpy import *
import numpy as np
from numpy.typing import NDArray
import scipy


import importlib.resources as pkg_resources  # python > 3.7
from designer import material_data
import designer.material_data.exp_eq as exp_eq

# use these wrapper functions to select which model / exp data to use
def get_n_SiO2(wl):
    return exp_eq.get_n_SiO2_Sellmeier(wl)

def get_n_SiO2_exp(wl):
    return get_SiO2_exp(wl)

def get_n_TiO2(wl):
    return exp_eq.get_n_TiO2_Sellmeier(wl)


def get_n_Si(wl):
    return get_Si_exp(wl)

def get_n_BK7(wl):
    return exp_eq.get_n_BK7_Sellmeier(wl)

def get_n_Air(wl):
    # approximate
    return 1.

def get_n_Ta2O5_xc(wl):
    return exp_eq.get_n_Ta2O5_Cauchy(wl)

def get_n_SiO2_xc(wl):
    return exp_eq.get_n_SiO2_Cauchy(wl)

def get_n_MgF2_xc(wl):
    return exp_eq.get_n_MgF2_Cauchy(wl)

def get_n_1(wl):
    return wl / wl  # broadcast if is instance of np.array


def get_n_1_5(wl):
    return 1.5 * wl / wl


def get_n_2(wl):
    return 2 * wl / wl


def get_n_free(wl, n: complex):
    return wl / wl * n




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


cached_SiO2 = False
wls_SiO2 = None
n_SiO2 = None
n_SiO2_interp = None


def get_SiO2_exp(wl):
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

    global cached_SiO2, wls_SiO2, n_SiO2
    if not cached_SiO2:
        wls_SiO2, n_SiO2 = load_from_file(
            'SiO2_n_Rodriguez-de_Marcos.csv',
            'SiO2_k_Rodriguez-de_Marcos.csv',
        )
        n_SiO2 = n_SiO2.real
        n_SiO2_interp = scipy.interpolate.interp1d(wls_SiO2, n_SiO2)

    return n_SiO2_interp(wl)
