from torch.autograd import Function, gradcheck
from utils import *


class CorrelationLoss(Function):
    """
    important tricks:
        1. regularize variance matrices
        2. "trace norm" is the trace of sqrt(T'T), instead of sqrt((T'T).trace())
    """
    @staticmethod
    def forward(ctx, H1, H2, sigma_reg):
        """
        :param ctx: context
        :param H1, H2: m * o tensors (m: batch size; o: output dimension)
        :param sigma_reg: regularizer on \Sigma_{11} and \Sigma{22}
        :return: corr coeff of H1 & H2
        """
        sample_size = H1.shape[0]  # basically the batch size

        # turn H into o * m
        H1 = torch.t(H1)
        H2 = torch.t(H2)
        H1_bar = center(H1)
        H2_bar = center(H2)

        # compute variance / covariance matrices
        var11 = regularized_variance(H1_bar, sigma_reg)
        var22 = regularized_variance(H2_bar, sigma_reg)
        covar12 = H1_bar.mm(H2_bar.t()) / (sample_size - 1)

        # form matrix T
        var11_rootinv = inverse_sqrt(var11)
        var22_rootinv = inverse_sqrt(var22)
        T = var11_rootinv.mm(covar12).mm(var22_rootinv)

        U, D, V = torch.svd(T)

        if ctx is not None:
            ctx.sample_size = sample_size
            ctx.save_for_backward(H1_bar, H2_bar, var11_rootinv, var22_rootinv, U, D.diag(), V)

        corr = mat_sqrt(T.t().mm(T)).trace()
        return corr  # supposed to be corr of H1 and H2???

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: has only one output
        :return:
        """
        H1_bar, H2_bar, var11_rootinv, var22_rootinv, U, D, V = ctx.saved_tensors

        delta12 = var11_rootinv.mm(U).mm(V.t()).mm(var22_rootinv)
        delta11 = - var11_rootinv.mm(U).mm(D).mm(U.t()).mm(var11_rootinv) / 2
        delta22 = - var22_rootinv.mm(V).mm(D).mm(V.t()).mm(var22_rootinv) / 2

        dfdH1 = (2 * delta11.mm(H1_bar) + delta12.mm(H2_bar)) / (ctx.sample_size - 1)
        dfdH2 = (2 * delta22.mm(H2_bar) + delta12.t().mm(H1_bar)) / (ctx.sample_size - 1)

        print("grad output: ", grad_output)
        return dfdH1.t(), dfdH2.t(), None


corrloss = CorrelationLoss.apply
