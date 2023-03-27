from ..preprocessing import PreprocessingBase
from ..trading import TraderBase


class SplitBacktester:
    def __init__(self, preprocessor: PreprocessingBase, trader: TraderBase):
        self._preprocessor = preprocessor
        self._trader = trader
        self._not_na_columns = None

    def fit(self, train_returns):
        self._not_na_columns = ~train_returns.isna().any(axis=0)
        train_returns = train_returns.loc[:, self._not_na_columns]

        preprocessed_train_returns = self._preprocessor.fit_transform(train_returns)
        self._trader.fit(preprocessed_train_returns)
        return self

    def backtest(self, test_returns, test_returns_fwd=None):
        test_returns = test_returns.loc[:, self._not_na_columns].fillna(0)

        if test_returns_fwd is None:
            test_returns_fwd = test_returns.shift(-1).iloc[:-1].copy()
            test_returns = test_returns.iloc[:-1].copy()

        trading_mask = self.get_trading_mask(test_returns)

        trading_returns = (test_returns_fwd * trading_mask).sum(axis=1)

        return trading_returns

    def get_trading_mask(self, test_returns):
        preprocessed_test_returns = self._preprocessor.fit_transform(test_returns)
        trading_mask = self._trader.transform(preprocessed_test_returns)
        return trading_mask
