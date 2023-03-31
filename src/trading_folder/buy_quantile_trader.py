from .trader_base import TraderBase
import numpy as np
import pandas as pd


class BuyQuantileTrader(TraderBase):
    def __init__(self, n_splits: int, d: int):
        if d < 1 or d > n_splits:
            raise ValueError(
                f'Value of d outside expected boundaries: d must be between 1 and {n_splits}, got {d} instead!')
        self._d = d
        self._n_splits = n_splits
        self._lower_q = (d - 1) / self._n_splits
        self._upper_q = d / self._n_splits

    def train(self, X_train: pd.DataFrame,
              y_train: pd.DataFrame = None) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def compute_trading_mask(self, X_test: pd.DataFrame):
        lower, upper = np.quantile(X_test, (self._lower_q, self._upper_q),
                                   axis=1).reshape((2, -1, 1))

        trading_mask = (X_test >= lower) & (X_test <= upper)
        trading_mask /= trading_mask.sum(axis=1).values.reshape((-1, 1))
        trading_mask = trading_mask.fillna(0)

        return trading_mask


BuyQuantileTradingRule = BuyQuantileTrader
