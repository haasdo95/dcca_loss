from torch.autograd import Function
from utils import *
import torch
from torch import nn

print("Using PyTorch Version: ", torch.__version__)


class CorrelationLoss(Function):
    """
    important tricks:
        1. regularize variance matrices
        2. "trace norm" is the trace of sqrt(T'T), instead of sqrt((T'T).trace())
    """
    @staticmethod
    def forward(ctx, H1, H2, sigma_reg, ledoit: bool):
        """
        :param ctx: context
        :param H1, H2: m * o tensors (m: batch size; o: output dimension)
        :param sigma_reg: regularizer on \Sigma_{11} and \Sigma{22}
        :param ledoit: True if using ledoit to estimate covariance
        :return: corr coeff of H1 & H2
        """
        # ledoit will give you the (pseudo)inverse directly: thus no need to combine inverse and sqrt
        sample_size = H1.shape[0]  # basically the batch size
        output_dim = H1.shape[1]

        # turn H into o * m
        H1 = torch.t(H1)
        H2 = torch.t(H2)
        H1_bar = center(H1)
        H2_bar = center(H2)

        # compute variance / covariance matrices
        if ctx is not None:
            ctx.ledoit = ledoit
        if ledoit:
            print("USING LEDOIT")
            var11, shrinkage_11 = covariance_matrix(H1_bar, None)
            var22, shrinkage_22 = covariance_matrix(H2_bar, None)
            assert not shrinkage_11.requires_grad
            assert not shrinkage_22.requires_grad
            if ctx is not None:
                ctx.shrinkage_11 = shrinkage_11
                ctx.shrinkage_22 = shrinkage_22
        else:
            var11 = covariance_matrix(H1_bar, sigma_reg)
            var22 = covariance_matrix(H2_bar, sigma_reg)
        covar12 = H1_bar.mm(H2_bar.t()) / float(sample_size)

        # form matrix T
        var11_rootinv = inverse_sqrt(var11)
        var22_rootinv = inverse_sqrt(var22)
        T = var11_rootinv.mm(covar12).mm(var22_rootinv)

        U, D, V = torch.svd(T)

        if ctx is not None:
            ctx.sample_size = sample_size
            ctx.output_dim = output_dim
            ctx.save_for_backward(H1_bar, H2_bar, var11_rootinv, var22_rootinv, U, D.diag(), V)

        corr = torch.sum(D)  # trace norm == sum of singular values
        return -corr  # minus sign here cuz you need to maximize corr

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: has only one output
        :return:
        """
        # print("backproping!")
        H1_bar, H2_bar, var11_rootinv, var22_rootinv, U, D, V = ctx.saved_tensors

        # TODO: DO THE MATH and figure out how to backprop with Ledoit

        delta12 = var11_rootinv.mm(U).mm(V.t()).mm(var22_rootinv)
        delta11 = - var11_rootinv.mm(U).mm(D).mm(U.t()).mm(var11_rootinv) / 2
        delta22 = - var22_rootinv.mm(V).mm(D).mm(V.t()).mm(var22_rootinv) / 2

        if ctx.ledoit:
            shrinkage_11 = ctx.shrinkage_11
            tr_11 = delta11.trace()
            delta11 = (1-shrinkage_11) * delta11
            delta11 += (shrinkage_11 * tr_11 / ctx.output_dim) * torch.eye(ctx.output_dim, dtype=delta11.dtype)

            shrinkage_22 = ctx.shrinkage_22
            tr_22 = delta22.trace()
            delta22 = (1-shrinkage_22) * delta22
            delta22 += (shrinkage_22 * tr_22 / ctx.output_dim) * torch.eye(ctx.output_dim, dtype=delta22.dtype)

        dfdH1 = (2 * delta11.mm(H1_bar) + delta12.mm(H2_bar)) / ctx.sample_size
        dfdH2 = (2 * delta22.mm(H2_bar) + delta12.t().mm(H1_bar)) / ctx.sample_size

        return -dfdH1.t(), -dfdH2.t(), None, None


CorrLoss = CorrelationLoss.apply


class CorrLayer(nn.Module):
    """
    implemented only to make ONNX graph work; DON'T use
    """
    def __init__(self, reg):
        super(CorrLayer, self).__init__()
        ledoit = False
        if reg is None:
            ledoit = True
        self.reg = reg
        self.ledoit = ledoit

    def forward(self, H1H2):
        H1 = H1H2[0]
        H2 = H1H2[1]
        return CorrelationLoss.forward(None, H1, H2, self.reg, self.ledoit)
