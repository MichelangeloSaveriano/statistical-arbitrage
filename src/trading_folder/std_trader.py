import pandas as pd

from .trader_base import TraderBase
import numpy as np


class StdTrader(TraderBase):
    def __init__(self, gamma=2):
        self._gamma = gamma
        self._thresholds = None

    def train(self, X_train: pd.DataFrame,
              y_train: pd.DataFrame = None):
        self._thresholds = X_train.std(axis=0, keepdims=True) * self._gamma

    def compute_trading_mask(self, X_test: pd.DataFrame) -> pd.DataFrame:
        data_matrix = X_test.values
        short_mask = (data_matrix >= self._thresholds).astype(float)
        long_mask = (data_matrix <= -self._thresholds).astype(float)

        short_mask /= np.fmax(1, short_mask.sum(axis=1).reshape((-1, 1)))
        long_mask /= np.fmax(1, long_mask.sum(axis=1).reshape((-1, 1)))

        trading_mask = long_mask - short_mask
        return pd.DataFrame(trading_mask, columns=X_test.columns, index=X_test.index)


StdTradingRule = StdTrader
