import unittest
import numpy as np
import sys
sys.path.append("./designer/script")

import tmm.mat_utils as mat

class TestMat(unittest.TestCase):
    def test_mul(self):
        m1 = np.random.random((2, 2))
        m2 = np.random.random((2, 2))
        m3 = m1.copy()
        mat.mul_right(m3, m2)
        self.assertAlmostEqual(np.dot(m1, m2)[0, 0], m3[0, 0])
        self.assertAlmostEqual(np.dot(m1, m2)[0, 1], m3[0, 1])
        self.assertAlmostEqual(np.dot(m1, m2)[1, 0], m3[1, 0])
        self.assertAlmostEqual(np.dot(m1, m2)[1, 1], m3[1, 1])

if __name__ == "__main__":
    unittest.main()