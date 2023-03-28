from .trader_base import TraderBase
import numpy as np


class StdTrader(TraderBase):
    def __init__(self, gamma=2):
        self._gamma = gamma
        self._thresholds = None

    def train(self, X_train, y_train=None):
        self._thresholds = X_train.std(axis=0, keepdims=True) * self._gamma

    def compute_trading_mask(self, X_test):
        short_mask = (X_test >= self._thresholds).astype(float)
        long_mask = (X_test <= -self._thresholds).astype(float)

        short_mask /= np.fmax(1, short_mask.sum(axis=1).reshape((-1, 1)))
        long_mask /= np.fmax(1, long_mask.sum(axis=1).reshape((-1, 1)))

        trading_mask = long_mask - short_mask
        return trading_mask


StdTradingRule = StdTrader
