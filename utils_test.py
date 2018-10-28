import unittest
import torch
from torch.autograd import Variable
from utils import *
import numpy as np


class TestUtils(unittest.TestCase):
    def test_centering(self):
        x = torch.normal(torch.ones((6, 12)) * 666, torch.ones((6, 12)))
        x_bar = center(x)
        self.assertEqual(x.shape, x_bar.shape)
        mean = torch.mean(x_bar, 1)
        zeros = torch.zeros_like(mean, dtype=mean.dtype)
        self.assertTrue(torch.all(torch.lt(torch.abs(torch.add(mean, -zeros)), 1e-3)))

    # Credit Goes to: Subhransu Maji (smaji@cs.umass.edu)
    # URL: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    # Cited on Sept. 30th 2018
    @staticmethod
    def create_spd(feature_dim, sample_dim, reg):
        pts = np.random.randn(sample_dim, feature_dim).astype(np.float32)
        sA = np.dot(pts.T, pts) / sample_dim + reg * np.eye(feature_dim).astype(np.float32)
        X = torch.from_numpy(sA)
        return X

    def multivariate_normal(self, feature_dim, sample_dim, reg):
        """
        :return: centered H with shape (o * m)
        """
        cov = self.create_spd(feature_dim, sample_dim, reg)
        print("singular values: ", torch.svd(cov)[1])
        return cov, np.stack(
            [
                np.random.multivariate_normal(np.zeros(feature_dim), cov, check_valid="warn", tol=10E-2)
                for _ in range(sample_dim)
            ]
        ).T

    def test_inverse_sqrt(self):
        def compute_error(A, sA):
            sA = sA.mm(sA)
            sA = sA.inverse()
            err = (A - sA).mean()
            return err
        spd = self.create_spd(12, 24, 0.1)
        invsqrt = inverse_sqrt(spd)
        err = float(compute_error(spd, invsqrt))
        print("Error: ", err)
        self.assertAlmostEqual(err, 0)

    def test_ledoit(self):
        """
        you could play around with shape to see how
        Ledoit estimator is consistently better than SCM
        """
        shape = (64, 32)  # o * m
        truth_cov, H = self.multivariate_normal(shape[0], shape[1], reg=0)
        H = Variable(torch.from_numpy(H), requires_grad=False)
        H_bar = center(H)
        ledoit_cov, precision, shrinkage = covariance_matrix(H_bar, reg=None)
        self.assertTrue(ledoit_cov.shape[0] == shape[0])
        self.assertTrue(np.allclose(torch.eye(shape[0]), ledoit_cov @ precision))
        print(shrinkage)
        print(truth_cov)
        truth_cov = truth_cov.numpy()
        scm_cov = (H_bar.mm(H_bar.t())).numpy()/(shape[1] - 1)

        norm_ord = "fro"
        ledoit_err = np.linalg.norm(truth_cov - ledoit_cov, ord=norm_ord)
        scm_err = np.linalg.norm(truth_cov - scm_cov, ord=norm_ord)
        print("ledoit compared to truth", ledoit_err)
        print("SCM compared to truth", scm_err)
        self.assertTrue(ledoit_err < scm_err)
