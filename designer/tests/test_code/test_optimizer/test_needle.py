import unittest
import numpy as np
import copy


import sys
sys.path.append('./designer/script/')
from film import FilmSimple
from optimizer.needle_insert import make_test_insert_film, insert_1_layer


class TestNeedle(unittest.TestCase):
    def test_make_insert_film(self):
        f = FilmSimple('SiO2', 'TiO2', 'SiO2', np.array([1., 2., 3.]))
        insert_idx = make_test_insert_film(f, 2)
        self.assertListEqual(list(f.get_d()), [0.5, 0, 0.5, 0., 0., 1, 0., 1, 0., 0., 1.5, 0., 1.5, 0., 0.])
        self.assertListEqual(insert_idx, [0, 2, 5, 7, 10, 12])

        f = FilmSimple('SiO2', 'TiO2', 'SiO2', np.array([3.]))
        insert_idx = make_test_insert_film(f, 2)
        self.assertListEqual(list(f.get_d()), [1.5, 0., 1.5, 0., 0.])
        self.assertListEqual(insert_idx, [0, 2])
        


if __name__ == '__main__':
    unittest.main()