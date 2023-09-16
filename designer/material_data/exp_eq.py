import numpy as np


def get_n_SiO2_Sellmeier(wl):
    wl = wl * 1e-3  # nm to \mu m
    # SiO2: Ghosh 1999 crystal, alpha-quartz
    # NOTE: applicable to 198 nm - 2050 nm
    return np.sqrt(1.28604141 + 1.07044083 * wl**2 /
                   (wl**2 - 0.0100585997) + 1.10202242 * wl**2 / (wl**2 - 100.))


def get_n_TiO2_Sellmeier(wl):
    wl = wl * 1e-3
    # TiO2: Devore 1951, crystal
    # NOTE: applicable to 430 nm - 1530 nm
    return np.sqrt(5.913 + 0.2441 / (wl**2 - 0.0803))


def get_n_Ta2O5_Cauchy(wl):
    wl = wl * 1e-3
    return 2.083033 + 3.0398531e-2 / wl ** 2 + 6.6997423e-9 / wl ** 4

def get_n_SiO2_Cauchy(wl):
    wl = wl * 1e-3
    return 1.476128 + 1.5048792e-3 / wl ** 2 + 4.3051470e-4 / wl ** 4

def get_n_MgF2_Cauchy(wl):
    wl = wl * 1e-3
    return 1.384 - 3.3651e-3 / wl ** 2+ 6.429e-4 / wl ** 4