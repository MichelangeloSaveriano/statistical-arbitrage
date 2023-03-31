from typing import Tuple, Any

import pandas as pd
import numpy as np
from pandas import DataFrame

from .trader_base import TraderBase, TradingRule
from ..laplacian_estimators import BaseLaplacianEstimator
from .quantiles_trader import QuantilesTradingRule


class SpreadsTrader(TraderBase):
    def __init__(self, laplacian_estimator: BaseLaplacianEstimator,
                 trading_rule: TradingRule | list[TradingRule] | dict[str, TradingRule] = None,
                 normalize_input=True):
        self._laplacian_estimator = laplacian_estimator
        self._L = None

        self._trading_rule = trading_rule
        if self._trading_rule is None:
            self._trading_rule = QuantilesTradingRule(normalize_input=False)
        elif isinstance(trading_rule, list):
            self._trading_rule = dict(zip(range(len(trading_rule)), trading_rule))

        self._normalize_input = normalize_input
        self._train_std = None

    def compute_spreads(self, train: pd.DataFrame,
                        test: pd.DataFrame = None) -> DataFrame | tuple[DataFrame, DataFrame | tuple[DataFrame]]:
        train_data = train.values.copy()

        if self._normalize_input:
            if self._train_std is None:
                self._train_std = train_data.std(axis=0,
                                                 keepdims=True)
            train_data /= self._train_std

        if self._L is None:
            self._L = self._laplacian_estimator.fit_laplacian(train_data)

        train_spreads = train_data @ self._L

        train_spreads = pd.DataFrame(train_spreads,
                                     index=train.index, columns=train.columns)

        if test is None:
            return train_spreads

        return train_spreads, self.compute_spreads(test)

    def train(self, X_train, y_train=None):
        train_spreads = self.compute_spreads(X_train)

        if isinstance(self._trading_rule, dict):
            for trader in self._trading_rule.values():
                trader.train(train_spreads)
        else:
            self._trading_rule.train(train_spreads)

        # self._trading_rule.train(train_spreads)

        return self

    def compute_trading_mask(self, X_test: pd.DataFrame) -> pd.DataFrame | dict[pd.DataFrame]:
        test_spreads = self.compute_spreads(X_test)

        if isinstance(self._trading_rule, dict):
            return {trader_name: trader.compute_trading_mask(test_spreads)
                    for trader_name, trader in self._trading_rule.items()}

        trading_mask = self._trading_rule.compute_trading_mask(test_spreads)
        return trading_mask
