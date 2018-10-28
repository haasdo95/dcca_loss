import torch
from torch.autograd import Variable
from typing import *
from sklearn.covariance import LedoitWolf


def inverse_sqrt(A: Union[Variable, torch.Tensor]):
    """
    :param A: SPD matrix
    :return: inverse sqrt of matrix X
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    U, S, V = A.svd()  # note: U.mm(V.t()) is always identity
    S_invsqrt = torch.sqrt(1 / S)
    sA = (V.mm(S_invsqrt.diag())).mm(U.t())
    return sA


def eigen_sqrt(A):
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    U, S, V = A.svd()  # note: U.mm(V.t()) is always identity
    S_sqrt = torch.sqrt(S)
    sA = (U.mm(S_sqrt.diag())).mm(U.t())
    return sA


def cholesky_decomposition(A):
    """
    CREDIT: https://www.pugetsystems.com/labs/hpc/PyTorch-for-Scientific-Computing---Quantum-Mechanics-Example-Part-3-Code-Optimizations---Batched-Matrix-Operations-Cholesky-Decomposition-and-Inverse-1225/#batched-cholesky-decomposition-and-matrix-inverse-in-pytorch
    this is an implementation of "The Cholesky–Banachiewicz algorithm"
    :return:
    """
    return torch.potrf(A, upper=False)


def center(H: Union[Variable, torch.Tensor]) -> Union[Variable, torch.Tensor]:
    """
    :param H: dimension o * m
    :return: centered H_bar
    """
    H_bar = (H.t() - torch.mean(H, 1)).t()
    return H_bar


def covariance_matrix(H_bar: Union[Variable, torch.Tensor], reg: float) -> Any:
    """
    :param H_bar: dimension o * m, centered
    :param reg: tunable hyper-parameter to make covmat invertible. None if using Ledoit & Wolf
    :return: cov mat of size o * o; or Tuple[covmat, covmat^{-1}, shrinkage] if Ledoit
    """
    if reg is not None:
        sample_size = H_bar.shape[1]
        out_dim = H_bar.shape[0]
        return H_bar.mm(H_bar.t()) / (sample_size - 1) + reg * torch.eye(out_dim, dtype=H_bar.dtype)
    else:  # using Ledoit estimator
        H_bar = H_bar.t()  # m * o
        ledoit = LedoitWolf(store_precision=True, assume_centered=False)
        ledoit.fit(H_bar)
        shrinkage = ledoit.shrinkage_
        covmat = ledoit.covariance_
        precision = ledoit.precision_  # estimated inverse of cov mat
        # need to compute: what is the equivalent "regularizer" here?
        return covmat, precision, shrinkage
