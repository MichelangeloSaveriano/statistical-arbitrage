import pandas as pd

from split_backtester import SplitBacktester

from ..preprocessing import PreprocessingBase
from ..trading import TradingRuleBase

import copy
from tsxv.splitTrain import split_train
from concurrent.futures import ThreadPoolExecutor


class ConfigBacktester:
    def __init__(self, preprocessor: PreprocessingBase, trader: TradingRuleBase,
                 train_window_size=60, split_window_size=12):
        self._preprocessor = preprocessor
        self._trader = trader
        self._train_window_size = train_window_size
        self._split_window_size = split_window_size
        self._split_backtesters = None

    def backtest(self, returns):
        train_idx, test_idx = split_train(returns.index.values, numInputs=self._train_window_size,
                                          numOutputs=self._split_window_size + 1, numJumps=1)
        train_idx = train_idx[::self._split_window_size]
        test_idx = test_idx[::self._split_window_size]

        executor = ThreadPoolExecutor()
        self._split_backtesters = list(
            executor.map(lambda idx: SplitBacktester(preprocessor=copy.deepcopy(self._preprocessor),
                                                     trader=copy.deepcopy(self._trader)).fit(returns.loc[idx]),
                         train_idx))

        return pd.concat([backtester.backtest(returns.loc[idx])
                          for idx, backtester in zip(test_idx, self._split_backtesters)])
