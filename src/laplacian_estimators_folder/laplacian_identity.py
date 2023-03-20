from .laplacian_estimator_base import BaseLaplacianEstimator
import numpy as np


class LaplacianIdentityLaplacianEstimator(BaseLaplacianEstimator):

    def __init__(self):
        self.L = None

    def fit_laplacian(self, train):
        L = np.eye(train.shape[1])
        self.L = L
        return L
