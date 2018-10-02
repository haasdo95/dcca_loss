import torch
from torch.autograd import Variable
from typing import *

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


def mat_sqrt(A: Union[Variable, torch.Tensor]):
    """
    :param A: SPD matrix
    :return: inverse sqrt of matrix X
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    U, S, V = A.svd()  # note: U.mm(V.t()) is always identity
    S_invsqrt = torch.sqrt(S)
    sA = (U.mm(S_invsqrt.diag())).mm(V.t())
    return sA


def center(H: Union[Variable, torch.Tensor]) -> Union[Variable, torch.Tensor]:
    """
    :param H: dimension o * m
    :return: centered H_bar
    """
    H_bar = (H.t() - torch.mean(H, 1)).t()
    return H_bar


def regularized_variance(H_bar: Union[Variable, torch.Tensor], reg: float) -> Union[Variable, torch.Tensor]:
    """
    :param H: dimension o * m
    :return: corr mat of size o * o
    """
    sample_size = H_bar.shape[1]
    out_dim = H_bar.shape[0]
    return H_bar.mm(H_bar.t()) / (sample_size - 1) + reg * torch.eye(out_dim, dtype=H_bar.dtype)
