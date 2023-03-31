from .laplacian_estimator_base import BaseLaplacianEstimator
from .graph_learning_algorithm import learn_connected_graph_heavy_tails
import numpy as np
import pandas as pd

class LaplacianGraphLearningLaplacianEstimator(BaseLaplacianEstimator):
    def __init__(self, distribution='gaussian', weights=None,
                 normalize=True, w0=None,
                 max_iter=1000, tol=5e-5,
                 d=1, nu=None, verbose=False,
                 use_abs=True,
                 laplacian_root=False, adjust_laplacian=False):

        self._weights = weights
        self._w0 = w0  # weight-init
        self._d = d  # Degree vector

        self._normalize = normalize

        self._max_iter = max_iter
        self._tol = tol

        self._distribution = distribution
        self._nu = nu
        if self._nu is None and self._distribution == 'student':
            self._nu = 3

        self._rho = 1
        self._update_rho = True
        self._mu = 2
        self._tau = 2

        self._use_abs = use_abs
        self._laplacian_root = laplacian_root
        self._adjust_laplacian = adjust_laplacian

        self._verbose = verbose

    def fit_laplacian(self, train):
        S = train.T @ train
        S_sign = np.ones_like(S)

        if self._use_abs:
            S_sign = np.sign(S)
            S *= S_sign

        results_learn_graph = learn_connected_graph_heavy_tails(train, heavy_type=self._distribution,
                                                                is_covariance=False, normalize=self._normalize,
                                                                nu=self._nu, w0=self._w0, d=self._d,
                                                                rho=self._rho, update_rho=self._update_rho,
                                                                max_iter=self._max_iter, tol=self._tol,
                                                                mu=self._mu, tau=self._tau,
                                                                verbose=self._verbose
                                                                )
        L = results_learn_graph['L']

        if self._weights == 'uniform':
            A = (np.diag(np.diag(L)) - L) > 1e-3
            A_weight = A / A.sum(axis=0, keepdims=True)
            L = np.eye(A_weight.shape[0]) - A_weight

        if self._laplacian_root:
            lambdas, V = np.linalg.eigh(L)
            L = V @ np.diag(np.sqrt(np.fmax(0, lambdas))) @ V.T

        if self._use_abs:
            L *= S_sign

        if self._adjust_laplacian:
            A = np.diag(np.diag(L)) - L
            train_pred = train @ A
            beta_ = (train * train_pred).sum(axis=0) / (train ** 2).sum(axis=0)
            L = (np.diag(1 / beta_) - A)
            L = np.diag(beta_) @ L

        return L
