import unittest
import torch
from utils import center, inverse_sqrt
import numpy as np


class TestUtils(unittest.TestCase):
    def test_centering(self):
        x = torch.normal(torch.ones((6, 12)) * 666, torch.ones((6, 12)))
        x_bar = center(x)
        self.assertEqual(x.shape, x_bar.shape)
        mean = torch.mean(x_bar, 1)
        zeros = torch.zeros_like(mean, dtype=mean.dtype)
        self.assertTrue(torch.all(torch.lt(torch.abs(torch.add(mean, -zeros)), 1e-3)))

    def test_inverse_sqrt(self):
        # Credit Goes to: Subhransu Maji (smaji@cs.umass.edu)
        # URL: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
        # Cited on Sept. 30th 2018
        def create_spd(dim1, dim2, tau):
            pts = np.random.randn(dim2, dim1).astype(np.float32)
            sA = np.dot(pts.T, pts) / dim2 + tau * np.eye(dim1).astype(np.float32)
            X = torch.from_numpy(sA)
            return X

        def compute_error(A, sA):
            sA = sA.mm(sA)
            sA = sA.inverse()
            err = (A - sA).mean()
            return err

        spd = create_spd(12, 24, 0.1)
        invsqrt = inverse_sqrt(spd)
        err = float(compute_error(spd, invsqrt))
        print("Error: ", err)
        self.assertAlmostEqual(err, 0)
