from .trader_base import TraderBase
import numpy as np
import pandas as pd


class QuantilesTrader(TraderBase):
    def __init__(self, q=0.2, normalize_input=True):
        self._q = q
        self._q_half = self._q / 2

        self._normalize_input = normalize_input
        self._train_std = None

    def train(self, X_train, y_train=None):
        if self._normalize_input:
            self._train_std = X_train.std(axis=0)

    def compute_trading_mask(self, X_test: pd.DataFrame):
        if self._normalize_input and self._train_std is not None:
            X_test /= self._train_std

        lower, upper = np.quantile(X_test, (self._q_half, 1 - self._q_half),
                                   axis=1).reshape((2, -1, 1))

        q_mask_short = X_test >= upper
        short_mask = q_mask_short
        short_mask /= short_mask.sum(axis=1).values.reshape((-1, 1))
        short_mask = short_mask.fillna(0)

        q_mask_long = X_test <= lower
        long_mask = q_mask_long
        long_mask /= long_mask.sum(axis=1).values.reshape((-1, 1))
        long_mask = long_mask.fillna(0)

        return long_mask - short_mask


QuantilesTradingRule = QuantilesTrader
