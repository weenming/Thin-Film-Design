import unittest
import numpy as np
import matplotlib.pyplot as plt
import copy
import time


import sys
sys.path.append('./designer/script/')
from film import TwoMaterialFilm
from spectrum import Spectrum
from optimizer.needle_insert import make_test_insert_film, insert_1_layer, get_insert_grad
from design import DesignSimple


class TestNeedle(unittest.TestCase):

    def test_make_insert_film(self):
        f = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', np.array([1., 2., 3.]))
        insert_idx = make_test_insert_film(f, 2)
        self.assertListEqual(list(f.get_d()), [0.5, 0, 0.5, 0., 0., 1, 0., 1, 0., 0., 1.5, 0., 1.5, 0., 0.])
        self.assertListEqual(insert_idx, [1, 3, 6, 8, 11, 13])

        f = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', np.array([3.]))
        insert_idx = make_test_insert_film(f, 2)
        self.assertListEqual(list(f.get_d()), [1.5, 0., 1.5, 0., 0.])
        self.assertListEqual(insert_idx, [1, 3])
        

    def test_insert_gd(self):
        
        # print('warm up:')
        # f = FilmSimple('SiO2', 'TiO2', 'SiO2', np.array([1., 2., 3.]))
        # target_spec_ls = [Spectrum(0., np.linspace(400, 1000, 500), np.ones(500, dtype='float'))]
        # grad = self.search_insert_helper(f, target_spec_ls, 10)
        # print('end of warm up')

        for search_pts in [10, 50]:

            f = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', np.array([100., 200., 300.]))
            target_spec_ls = [Spectrum(0., np.linspace(400, 1000, 500), np.ones(500, dtype='float'))]
            idx, grad = self.search_insert_helper(f, target_spec_ls, search_pts)

            d_depth = np.array([f.get_d()[:i].sum() for i in range(f.get_layer_number())])
            # plt.plot(d_depth[idx], grad[idx])
            # plt.show()


    def search_insert_helper(self, f: TwoMaterialFilm, target_ls, search_pts):
        layer_before = f.get_layer_number()
        t1 = time.time()
        idx = make_test_insert_film(f, search_pts)
        grad = get_insert_grad(f, target_ls)
        t2 = time.time()
        print(f'time to search {search_pts} pts in {layer_before} layers: takes {t2 - t1} s')
        return idx, grad




if __name__ == '__main__':
    unittest.main()
    