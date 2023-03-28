import pandas as pd
import numpy as np
from .trader_base import TraderBase
from ..laplacian_estimators import BaseLaplacianEstimator


class SpreadsTrader(TraderBase):
    def __init__(self, laplacian_estimator: BaseLaplacianEstimator, normalize_input=True, q=0.2):
        self._laplacian_estimator = laplacian_estimator
        self._L = None

        self._normalize_input = normalize_input
        self._train_std = None

        self._q = q
        self._q_half = self._q / 2
        self._lower = None
        self._upper = None

    def compute_spreads(self, train, test=None):
        train_data = train.values.copy()

        if self._normalize_input:
            if self._train_std is None:
                self._train_std = train_data.std(axis=0,
                                                 keepdims=True)
            train_data /= self._train_std

        if self._L is None:
            self._L = self._laplacian_estimator.fit_laplacian(train)

        train_spreads = train_data @ self._L
        if isinstance(train, pd.DataFrame):
            train_spreads = pd.DataFrame(train_spreads,
                                         index=train.index, columns=train.columns)

        if test is None:
            return train_spreads

        return train_spreads, self.compute_spreads(test)

    def train(self, X_train, y_train=None):
        train_spreads = self.compute_spreads(X_train)
        lower, upper = np.quantile(train_spreads, (self._q_half, 1 - self._q_half),
                                   axis=0).reshape((2, 1, -1))

        self._lower = lower
        self._upper = upper

        return self

    def compute_trading_mask(self, X_test):
        test_spreads = self.compute_spreads(X_test)

        lower, upper = np.quantile(test_spreads, (self._q_half, 1 - self._q_half),
                                   axis=1).reshape((2, -1, 1))

        q_mask_short = test_spreads >= upper
        # q_mask_short = test_spreads >= self._upper
        short_mask = q_mask_short
        short_mask /= short_mask.sum(axis=1).values.reshape((-1, 1))
        short_mask = short_mask.fillna(0)

        q_mask_long = test_spreads <= lower
        # q_mask_long = test_spreads <= self._lower
        long_mask = q_mask_long
        long_mask /= long_mask.sum(axis=1).values.reshape((-1, 1))
        long_mask = long_mask.fillna(0)

        return long_mask - short_mask

    def set_q(self, q):
        self._q = q
        self._q_half = self._q / 2
        self._lower = None
        self._upper = None
