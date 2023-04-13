import sys
sys.path.append("./designer/script")
sys.path.append('./')

import unittest
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import get_n



class GetNTest(unittest.TestCase):
    def plot(wls, n):
        plt.plot(wls, n.real, label='real')
        plt.plot(wls, n.imag, label='imag')
        plt.legend()
        plt.show()


    def test_from_file_Si(self, plot=False):
        # load file
        t_before_load = time.time()
        get_n.get_n_Si(500)
        t_after_load = time.time()

        N_wls = 1000
        wls = np.linspace(500, 1000, N_wls)
        n = np.empty(N_wls, dtype='complex')

        t_before_interp = time.time()
        n = get_n.get_n_Si(wls)
        t_after_interp = time.time()

        t_before_interp_onr_by_one = time.time()
        for i, wl in enumerate(wls):
            n[i] = get_n.get_n_Si(wl)
        t_after_interp_onr_by_one = time.time()

        if plot:
            self.plot(wls, n)

        print(f'load file: {t_after_load - t_before_load}s')
        print(f'interpolate n: {(t_after_interp - t_before_interp)}s for {N_wls} wl')
        print(f'interpolate n: {(t_after_interp_onr_by_one - t_before_interp_onr_by_one)}s for {N_wls} wl, repeat calling get_n')
        self.assertAlmostEqual(n[0], 4.294 - 1j * 0.044165)

    def test_calc_SiO2(self, plot=False):
        N_wls = 1000
        wls = np.linspace(500, 1000, N_wls)
        n = np.empty(N_wls, dtype='complex')

        t_before_calc = time.time()
        n = get_n.get_n_SiO2(wls)
        t_after_calc = time.time()

        t_before_calc_onr_by_one = time.time()
        for i, wl in enumerate(wls):
            n[i] = get_n.get_n_SiO2(wl)
        t_after_calc_onr_by_one = time.time()

        if plot:
            self.plot(wls, n)
        
        print(f'calculate n: {(t_after_calc - t_before_calc)}s for {N_wls} wl')
        print(f'calculate n: {(t_after_calc_onr_by_one - t_before_calc_onr_by_one)}s for {N_wls} wl, repeat calling get_n')




if __name__ == '__main__':
    unittest.main()