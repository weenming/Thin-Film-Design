from numpy import *
import cmath


def get_n_SiO2(wl):
    wl = wl * 1e-3 # nm to \mu m
    # SiO2: Ghosh 1999 crystal, alpha-quartz
    return cmath.sqrt(1.28604141 + 1.07044083 * wl**2 /
                  (wl**2 - 0.0100585997) + 1.10202242 * wl**2 / (wl**2 - 100.))

def get_n_TiO2(wl):
    wl = wl * 1e-3
    # TiO2: Devore 1951 crystal
    return cmath.sqrt(5.913+0.2441/(wl**2 - 0.0803))

def get_n_Air(wl):
    return 1.