import pandas as pd

from .split_backtester import SplitBacktester

from ..preprocessing import PreprocessingBase
from ..trading import TraderBase

import copy
from tsxv.splitTrain import split_train
from concurrent.futures import ThreadPoolExecutor


class ConfigBacktester:
    def __init__(self, preprocessor: PreprocessingBase, trader: TraderBase,
                 train_window_size=60, split_window_size=12):
        self._preprocessor = preprocessor
        self._trader = trader
        self._train_window_size = train_window_size
        self._split_window_size = split_window_size
        self._split_backtesters = None

    def fit(self, returns):
        train_idx, test_idx = split_train(returns.index.values, numInputs=self._train_window_size,
                                          numOutputs=self._split_window_size + 1, numJumps=self._split_window_size)

        executor = ThreadPoolExecutor()
        self._split_backtesters = list(
            executor.map(lambda idx: SplitBacktester(preprocessor=copy.deepcopy(self._preprocessor),
                                                     trader=copy.deepcopy(self._trader)).fit(returns.loc[idx]),
                         train_idx))

        return self

    def backtest(self, returns):
        train_idx, test_idx = split_train(returns.index.values, numInputs=self._train_window_size,
                                          numOutputs=self._split_window_size + 1, numJumps=self._split_window_size)

        strategy_returns = [backtester.backtest(returns.loc[idx])
                            for idx, backtester in zip(test_idx, self._split_backtesters)]
        return pd.concat(strategy_returns)

        # return pd.concat()
