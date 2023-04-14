import unittest
import numpy as np
import copy


import sys
sys.path.append('./designer/script/')
from film import TwoMaterialFilm


class TestFilm(unittest.TestCase):

    def remove_helper(self, input, out_expect):
        d = np.array(input)
        f = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2', d)
        f.remove_negative_thickness_layer()
        self.assertTrue(np.array_equal(f.get_d(), np.array(out_expect)))

    def test_delete_layer(self):
        self.remove_helper([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
        self.remove_helper([1, 0, 0, 4, 5, 6], [1, 4, 5, 6])
        self.remove_helper([1, 0, 0, 0, 5, 6], [6, 6])
        self.remove_helper([0, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6])
        self.remove_helper([1, 2, 3, 4, 0, 6], [1, 2, 3, 10])
        self.remove_helper([1, 2, 3, 4, 0, 0], [1, 2, 3, 4])
        self.remove_helper([1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5])

    def test_insert(self):
        f = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2',
                            np.array([1, 2, 3], dtype='float'))

        f_tmp = copy.deepcopy(f)
        self.assertRaises(
            AssertionError, lambda: f_tmp.insert_layer(0, -0.5, 100))
        f_tmp = copy.deepcopy(f)
        self.assertRaises(
            AssertionError, lambda: f_tmp.insert_layer(3, 0.5, 100))

        f_tmp = copy.deepcopy(f)
        f_tmp.insert_layer(0, 0.5, 100)
        self.assertListEqual(list(f_tmp.get_d()), [0.5, 100, 0.5, 2, 3])

        f_tmp = copy.deepcopy(f)
        f_tmp.insert_layer(0, 0.0, 100)
        self.assertListEqual(list(f_tmp.get_d()), [0.0, 100, 1., 2, 3])

        f_tmp = copy.deepcopy(f)
        f_tmp.insert_layer(0, 1., 100)
        self.assertListEqual(list(f_tmp.get_d()), [1., 100, 0., 2, 3])

        # single layer
        f = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2',
                            np.array([1], dtype='float'))

        f_tmp = copy.deepcopy(f)
        f_tmp.insert_layer(0, 0.5, 100)
        self.assertListEqual(list(f_tmp.get_d()), [0.5, 100, 0.5])

        f_tmp = copy.deepcopy(f)
        f_tmp.insert_layer(0, 0.0, 100)
        self.assertListEqual(list(f_tmp.get_d()), [0.0, 100, 1.])

        f_tmp = copy.deepcopy(f)
        f_tmp.insert_layer(0, 1., 100)
        self.assertListEqual(list(f_tmp.get_d()), [1., 100, 0.])


if __name__ == "__main__":
    unittest.main()
