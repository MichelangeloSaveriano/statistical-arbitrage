from laplacian_estimator_base import BaseLaplacianEstimator
import numpy as np


class LaplacianCorrKLaplacianEstimator(BaseLaplacianEstimator):

    def __init__(self, k, use_abs=True, weights='equal'):
        self._K = k
        self._use_abs = use_abs
        self._weights = weights

    def fit_laplacian(self, train):
        p = train.shape[1]
        S = np.corrcoef(train, rowvar=False)
        S_sign = np.ones_like(S)

        if self._use_abs:
            S_sign = np.sign(S)
            S *= S_sign

        L = np.eye(p)
        L[np.hstack([np.arange(p).reshape((-1, 1))] * self._K),
          np.argsort(S)[:, -(self._K + 1):-1]] = -1 / self._K

        if self._use_abs:
            L *= S_sign

        L = L.T
        L_sqrt = L
        return L, L_sqrt
