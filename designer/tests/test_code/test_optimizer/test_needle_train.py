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


class TestNeedleTrain(unittest.TestCase):

    def test_train_needle(self):

        f = TwoMaterialFilm('SiO2', 'TiO2', 'SiO2',
                            np.array([1000.], dtype='float'))
        target_spec_ls = [Spectrum(0., np.linspace(
            400, 1000, 500), np.ones(500, dtype='float'))]

        design = DesignSimple(target_spec_ls, f)
        design.TFNN_train(50)


if __name__ == '__main__':
    unittest.main()
