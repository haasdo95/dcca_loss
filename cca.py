from torch.autograd import Function, gradcheck
from utils import *


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

        # turn H into o * m
        H1 = torch.t(H1)
        H2 = torch.t(H2)
        H1_bar = center(H1)
        H2_bar = center(H2)

        # compute variance / covariance matrices
        if ledoit:
            var11, var11_inv, shrinkage_11 = covariance_matrix(H1, None)
            var22, var22_inv, shrinkage_22 = covariance_matrix(H2, None)
        else:
            var11 = covariance_matrix(H1_bar, sigma_reg)
            var22 = covariance_matrix(H2_bar, sigma_reg)
        covar12 = H1_bar.mm(H2_bar.t()) / (sample_size - 1)

        # form matrix T
        var11_rootinv = inverse_sqrt(var11)
        var22_rootinv = inverse_sqrt(var22)
        T = var11_rootinv.mm(covar12).mm(var22_rootinv)

        U, D, V = torch.svd(T)

        if ctx is not None:
            ctx.sample_size = sample_size
            ctx.save_for_backward(H1_bar, H2_bar, var11_rootinv, var22_rootinv, U, D.diag(), V)

        corr = eigen_sqrt(T.t().mm(T)).trace()  # ???
        return -corr  # minus sign here cuz you need to maximize corr

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: has only one output
        :return:
        """
        # print("backproping!")
        H1_bar, H2_bar, var11_rootinv, var22_rootinv, U, D, V = ctx.saved_tensors

        delta12 = var11_rootinv.mm(U).mm(V.t()).mm(var22_rootinv)
        delta11 = - var11_rootinv.mm(U).mm(D).mm(U.t()).mm(var11_rootinv) / 2
        delta22 = - var22_rootinv.mm(V).mm(D).mm(V.t()).mm(var22_rootinv) / 2

        dfdH1 = (2 * delta11.mm(H1_bar) + delta12.mm(H2_bar)) / (ctx.sample_size - 1)
        dfdH2 = (2 * delta22.mm(H2_bar) + delta12.t().mm(H1_bar)) / (ctx.sample_size - 1)

        return -dfdH1.t(), -dfdH2.t(), None, None


corrloss = CorrelationLoss.apply
