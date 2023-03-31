from abc import ABC, abstractmethod

import pandas as pd


class TraderBase(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame,
              y_train: pd.DataFrame = None) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def compute_trading_mask(self, X_test: pd.DataFrame) -> pd.DataFrame:
        pass


TradingRule = TraderBase


