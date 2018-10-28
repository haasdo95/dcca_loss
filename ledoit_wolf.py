# CITE: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/covariance/shrunk_covariance_.py#L328
import numpy as np
import torch


def ledoit_wolf_cov(X_bar):
    """
    :param X_bar: shape (o * m), not initially batch-major
    :return:
    """
    emp_cov = X_bar.mm(X_bar.t()) / X_bar.shape[1]
    shrinkage, mu = ledoit_wolf_shrinkage(X_bar, emp_cov)
    return (1 - shrinkage) * emp_cov + shrinkage * mu * torch.eye(X_bar.shape[0], dtype=emp_cov.dtype), shrinkage


def ledoit_wolf_shrinkage(X_bar, emp_cov):
    """
    :param X_bar: not initially batch-major
    :return: (shrinkage, mu)
    """
    with torch.no_grad():
        # A simple implementation of the formulas from Ledoit & Wolf
        n_features, n_samples = X_bar.shape
        X_bar = X_bar.t()  # batch major!
        mu = torch.trace(emp_cov) / n_features
        delta_ = emp_cov.clone()
        delta_.view(-1)[::n_features + 1] -= mu
        delta = (delta_ ** 2).sum() / n_features
        X2 = X_bar ** 2
        beta_ = 1. / (n_features * n_samples) * torch.sum(torch.mm(X2.t(), X2) / n_samples - emp_cov ** 2)
        beta = min(beta_, delta)
        shrinkage = beta / delta
        return float(shrinkage), mu


# DON'T use it: just a sanity-check
def ledoit_wolf_shrinkage_numpy(X, assume_centered=False, block_size=24):
    X = np.asarray(X)

    n_samples, n_features = X.shape

    # optionally center data
    if not assume_centered:
        X = X - X.mean(0)

    # A non-blocked version of the computation is present in the tests
    # in tests/test_covariance.py

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2 = X ** 2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_ = 0.  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += np.sum(np.dot(X2.T[rows], X2[:, cols]))
            delta_ += np.sum(np.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += np.sum(np.dot(X2.T[rows], X2[:, block_size * n_splits:]))
        delta_ += np.sum(
            np.dot(X.T[rows], X[:, block_size * n_splits:]) ** 2)
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += np.sum(np.dot(X2.T[block_size * n_splits:], X2[:, cols]))
        delta_ += np.sum(
            np.dot(X.T[block_size * n_splits:], X[:, cols]) ** 2)
    delta_ += np.sum(np.dot(X.T[block_size * n_splits:],
                            X[:, block_size * n_splits:]) ** 2)
    delta_ /= n_samples ** 2
    beta_ += np.sum(np.dot(X2.T[block_size * n_splits:],
                           X2[:, block_size * n_splits:]))
    # use delta_ to compute beta
    beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2. * mu * emp_cov_trace.sum() + n_features * mu ** 2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which whould invert
    # the value of covariances
    beta = min(beta, delta)
    # finally get shrinkage
    shrinkage = 0 if beta == 0 else beta / delta
    return shrinkage