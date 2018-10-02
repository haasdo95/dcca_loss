from cca import CorrelationLoss, corrloss
import torch
from torch.autograd import Variable
import numpy as np
import unittest


class TestCCA(unittest.TestCase):
    def test_cca(self):
        fwd_func = CorrelationLoss.forward

        shape = (120, 60)

        H1 = Variable(torch.randn(shape[0], shape[1], dtype=torch.double), requires_grad=True)
        H2 = Variable(torch.randn(shape[0], shape[1], dtype=torch.double), requires_grad=True)
        reg = 0.1

        print("1. Autograd")
        print("FORWARD: ")
        corr = fwd_func(None, H1, H2, reg)  # using autograd
        print("BACKWARD")
        corr.backward()
        # print("GRAD ON H1: ", H1.grad)
        # print("GRAD ON H2: ", H2.grad)
        H1_grad_auto = np.copy(H1.grad.data)
        H2_grad_auto = np.copy(H2.grad.data)

        H1.grad.data.zero_()
        H2.grad.data.zero_()

        print("2. MyGrad")
        print("FORWARD: ")
        corr = corrloss(H1, H2, reg)  # using my forward & backward
        print("BACKWARD")
        corr.backward()
        # print("GRAD ON H1: ", H1.grad)
        # print("GRAD ON H2: ", H2.grad)
        H1_grad_my = np.copy(H1.grad.data)
        H2_grad_my = np.copy(H2.grad.data)

        self.assertTrue(np.allclose(H1_grad_auto, H1_grad_my))
        self.assertTrue(np.allclose(H2_grad_auto, H2_grad_my))
