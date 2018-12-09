from torch.autograd import Function
from .utils import *
import torch

print("Using PyTorch Version: ", torch.__version__)


class CorrelationLoss(Function):
    """
    important tricks:
        1. regularize variance matrices
        2. "trace norm" is the trace of sqrt(T'T), instead of sqrt((T'T).trace())
    """
    @staticmethod
    def forward(ctx, H1, H2, sigma_reg, ledoit: bool, mu_gradient: bool):
        """
        :param ctx: context
        :param H1, H2: m * o tensors (m: batch size; o: output dimension)
        :param sigma_reg: regularizer on \Sigma_{11} and \Sigma{22}
        :param ledoit: True if using ledoit to estimate covariance
        :param mu_gradient: True if want gradient flow through mu
        :param semiledoit: use ledoit for forward prop only
        :param verbose: True if you want CCA to return lots of stuff
        :return: corr coeff of H1 & H2
        """
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
            ctx.mu_gradient = mu_gradient
        if ledoit:
            var11, shrinkage_11, mu_11 = covariance_matrix(H1_bar, None, mu_gradient)
            var22, shrinkage_22, mu_22 = covariance_matrix(H2_bar, None, mu_gradient)
            assert not shrinkage_11.requires_grad
            assert not shrinkage_22.requires_grad
            if ctx is not None:
                ctx.shrinkage_11 = shrinkage_11
                ctx.shrinkage_22 = shrinkage_22
        else:
            var11 = covariance_matrix(H1_bar, sigma_reg, None)
            var22 = covariance_matrix(H2_bar, sigma_reg, None)
        covar12 = H1_bar.mm(H2_bar.t()) / sample_size

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
        retval = -corr
        if ledoit:
            retval = -corr, shrinkage_11, shrinkage_22  # return everything to be crystal clear
        return retval  # minus sign here cuz you need to maximize corr

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        :param grad_outputs: never used since this is the loss function
        :return:
        """
        # print("backproping!")
        H1_bar, H2_bar, var11_rootinv, var22_rootinv, U, D, V = ctx.saved_tensors

        delta12 = var11_rootinv.mm(U).mm(V.t()).mm(var22_rootinv)
        delta11 = - var11_rootinv.mm(U).mm(D).mm(U.t()).mm(var11_rootinv) / 2
        delta22 = - var22_rootinv.mm(V).mm(D).mm(V.t()).mm(var22_rootinv) / 2

        if ctx.ledoit:
            shrinkage_11 = ctx.shrinkage_11
            tr_11 = delta11.trace()
            delta11 = (1-shrinkage_11) * delta11
            if ctx.mu_gradient:
                options = {"dtype": delta11.dtype, "device": delta11.device}
                delta11 += (shrinkage_11 * tr_11 / ctx.output_dim) * torch.eye(ctx.output_dim, **options)

            shrinkage_22 = ctx.shrinkage_22
            tr_22 = delta22.trace()
            delta22 = (1-shrinkage_22) * delta22
            if ctx.mu_gradient:
                options = {"dtype": delta22.dtype, "device": delta22.device}
                delta22 += (shrinkage_22 * tr_22 / ctx.output_dim) * torch.eye(ctx.output_dim, **options)

        dfdH1 = (2 * delta11.mm(H1_bar) + delta12.mm(H2_bar)) / ctx.sample_size
        dfdH2 = (2 * delta22.mm(H2_bar) + delta12.t().mm(H1_bar)) / ctx.sample_size

        return -dfdH1.t(), -dfdH2.t(), None, None, None


CorrLoss = CorrelationLoss.apply
