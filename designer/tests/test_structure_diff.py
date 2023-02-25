import unittest
import numpy as np
import sys
sys.path.append("./designer/script")
import utils.structure as s


class TestFilm(unittest.TestCase):
    def test_1_layer_equal(self):
        d1 = np.array([1])
        d2 = np.array([1])
        self.assertAlmostEqual(
            s._calculate_structure_difference_simple_film(
                d1, 1, None, d2, 0, None, None
            ), 1
        )

    def test_1_layer_unequal(self):
        d1 = np.array([2])
        d2 = np.array([1])
        self.assertAlmostEqual(
            s._calculate_structure_difference_simple_film(
                d1, 1, None, d2, 0, None, 10
            ), 10
        )

    def test_multi_layer_equal_layer_number(self):
        d1 = np.array([1, 2, 3])
        d2 = np.array([3, 1, 1])
        self.assertAlmostEqual(
            s._calculate_structure_difference_simple_film(
                d1, 1, 2, d2, 0, 1, 10
            ), 15
        )

    def test_multi_layer_unequal_layer_number(self):
        d1 = np.array([1, 2, 3, 4])
        d2 = np.array([3, 1, 5])
        self.assertAlmostEqual(
            s._calculate_structure_difference_simple_film(
                d1, 1, 2, d2, 0, 1, 10
            ), 21
        )


if __name__ == '__main__':
    unittest.main()
