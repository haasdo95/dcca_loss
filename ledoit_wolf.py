# CITE: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/covariance/shrunk_covariance_.py#L328
import numpy as np
import warnings
import torch


def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.
    Read more in the :ref:`User Guide <shrunk_covariance>`.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.
    assume_centered : bool
        If True, data are not centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data are centered before computation.
    block_size : int
        Size of the blocks into which the covariance matrix will be split.
    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    Notes
    -----
    The regularized (shrunk) covariance is:
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    where mu = trace(cov) / n_features
    """
    with torch.no_grad():
        X = np.asarray(X)
        # for only one feature, the result is the same whatever the shrinkage
        if len(X.shape) == 2 and X.shape[1] == 1:
            return 0.
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))

        if X.shape[0] == 1:
            warnings.warn("Only one sample available. "
                          "You may want to reshape your data array")
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