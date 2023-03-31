import pandas as pd
from typing import Self
from ..preprocessing import PreprocessingBase
from ..trading import TraderBase


class SplitBacktester:
    def __init__(self, preprocessor: PreprocessingBase, trader: TraderBase):
        self._preprocessor = preprocessor
        self._trader = trader
        self._not_na_columns = None

    def fit(self, train_returns: pd.DataFrame) -> Self:
        self._not_na_columns = ~train_returns.isna().any(axis=0)
        train_returns = train_returns.loc[:, self._not_na_columns]

        preprocessed_train_returns = self._preprocessor.fit_transform(train_returns)

        # if isinstance(self._trader, list):
        #     for trader in self._trader:
        #         trader.train(preprocessed_train_returns)
        # elif isinstance(self._trader, dict):
        #     for trader in self._trader.values():
        #         trader.train(preprocessed_train_returns)
        # else:
        #     self._trader.train(preprocessed_train_returns)

        self._trader.train(preprocessed_train_returns)

        return self

    def backtest(self, test_returns: pd.DataFrame,
                 test_returns_fwd: pd.DataFrame = None) -> pd.DataFrame:
        test_returns = test_returns.loc[:, self._not_na_columns].fillna(0)

        if test_returns_fwd is None:
            test_returns_fwd = test_returns.shift(-1).iloc[:-1].copy()
            test_returns = test_returns.iloc[:-1].copy()

        trading_mask = self.get_trading_mask(test_returns)

        if isinstance(trading_mask, list):
            return pd.DataFrame([(test_returns_fwd * mask).sum(axis=1) for mask in trading_mask]).T

        if isinstance(trading_mask, dict):
            return pd.DataFrame({trading_rule_name: (test_returns_fwd * mask).sum(axis=1)
                                 for trading_rule_name, mask in trading_mask.items()})

        trading_returns = (test_returns_fwd * trading_mask).sum(axis=1)
        # print(type(trading_returns))

        return pd.DataFrame(trading_returns).rename_axis(columns='TradingRule')

    def get_trading_mask(self,
                         test_returns: pd.DataFrame) -> pd.DataFrame:
        preprocessed_test_returns = self._preprocessor.fit_transform(test_returns)

        # if isinstance(self._trader, list):
        #     return [trader.compute_trading_mask(preprocessed_test_returns) for trader in self._trader]
        # if isinstance(self._trader, dict):
        #     return {trader_name: trader.compute_trading_mask(preprocessed_test_returns)
        #             for trader_name, trader in self._trader.items()}

        trading_mask = self._trader.compute_trading_mask(preprocessed_test_returns)
        return trading_mask
