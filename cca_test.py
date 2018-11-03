from cca import CorrelationLoss, CorrLoss
import torch
from torch.autograd import Variable
import numpy as np
import unittest
from time import time


class TestCCA(unittest.TestCase):
    def test_cca(self):
        """
        NOTE: test will pass only if shape[0] > shape[1]...
        """
        shape = (120, 60)
        H1 = torch.randn(shape[0], shape[1], dtype=torch.double, requires_grad=True)
        H2 = torch.randn(shape[0], shape[1], dtype=torch.double, requires_grad=True)
        reg = 0.1

        fwd_func = CorrelationLoss.forward
        start = time()
        corr = fwd_func(None, H1, H2, reg, False)  # using autograd
        corr.backward()
        print("autograd time taken", time() - start)
        H1_grad_auto = np.copy(H1.grad.data)
        H2_grad_auto = np.copy(H2.grad.data)
        H1.grad.data.zero_()
        H2.grad.data.zero_()

        start = time()
        corr = CorrLoss(H1, H2, reg, False)  # using my forward & backward
        corr.backward()
        print("my grad time taken", time() - start)
        H1_grad_my = np.copy(H1.grad.data)
        H2_grad_my = np.copy(H2.grad.data)
        self.assertTrue(np.allclose(H1_grad_auto, H1_grad_my))
        self.assertTrue(np.allclose(H2_grad_auto, H2_grad_my))

    def test_cca_ledoit(self):
        """
        NOTE: won't pass before I figure out how to backprop with Ledoit
        NOTE: there still seems to be funky numerical issues
        """
        shape = (4, 4)
        H1 = torch.randn(shape[0], shape[1], dtype=torch.double, requires_grad=True)
        H2 = torch.randn(shape[0], shape[1], dtype=torch.double, requires_grad=True)

        fwd_func = CorrelationLoss.forward
        start = time()
        corr = fwd_func(None, H1, H2, None, True)  # using autograd
        corr.backward()
        print("autograd time taken", time() - start)
        H1_grad_auto = np.copy(H1.grad.data)
        H2_grad_auto = np.copy(H2.grad.data)
        H1.grad.data.zero_()
        H2.grad.data.zero_()

        start = time()
        corr = CorrLoss(H1, H2, None, True)  # using my forward & backward
        corr.backward()
        print("my grad time taken", time() - start)
        H1_grad_my = np.copy(H1.grad.data)
        H2_grad_my = np.copy(H2.grad.data)

        print("auto: ", H1_grad_auto)
        print("my: ", H1_grad_my)

        print("auto - my: ", H1_grad_auto - H1_grad_my)

        self.assertTrue(np.allclose(H1_grad_auto, H1_grad_my, atol=10E-3))
        self.assertTrue(np.allclose(H2_grad_auto, H2_grad_my, atol=10E-3))

    def test_cca_speed(self):
        """
        customized gradient is faster than autograd
        takes only about two third of the time on my laptop
        """
        shape = (64, 64)
        H1 = Variable(torch.randn(shape[0], shape[1], dtype=torch.double), requires_grad=True)
        H2 = Variable(torch.randn(shape[0], shape[1], dtype=torch.double), requires_grad=True)
        reg = 0.1
        N = 100

        fwd_func = CorrelationLoss.forward
        start = time()
        for _ in range(N):
            corr = fwd_func(None, H1, H2, reg, False)  # using autograd
            corr.backward()
        print("autograd time taken", time() - start)

        start = time()
        for _ in range(N):
            corr = CorrLoss(H1, H2, reg, False)  # using my forward & backward
            corr.backward()
        print("my grad time taken", time() - start)
