import torch
from torch.autograd import Variable
from typing import *
from .ledoit_wolf import ledoit_wolf_cov


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
    U, S, _ = A.svd()  # note: U.mm(V.t()) is always identity
    S_sqrt = torch.sqrt(S)
    sA = (U.mm(S_sqrt.diag())).mm(U.t())
    return sA


def center(H: Union[Variable, torch.Tensor]) -> Union[Variable, torch.Tensor]:
    """
    :param H: dimension o * m
    :return: centered H_bar
    """
    H_bar = (H.t() - torch.mean(H, 1)).t()
    return H_bar


def covariance_matrix(H_bar: Union[Variable, torch.Tensor], reg: float, mu_gradient: bool) -> Any:
    """
    :param H_bar: dimension o * m, centered
    :param reg: tunable hyper-parameter to make covmat invertible. None if using Ledoit & Wolf
    :param mu_gradient: True if you want gradient to flow through mu
    :return: cov mat of size o * o; or Tuple[covmat, shrinkage] if Ledoit
    """
    if reg is not None:
        sample_size = H_bar.shape[1]
        out_dim = H_bar.shape[0]
        options = {"dtype": H_bar.dtype, "device": H_bar.device}
        return H_bar.mm(H_bar.t()) / float(sample_size) + reg * torch.eye(out_dim, **options)
    else:  # using Ledoit estimator
        return ledoit_wolf_cov(H_bar, mu_gradient)
