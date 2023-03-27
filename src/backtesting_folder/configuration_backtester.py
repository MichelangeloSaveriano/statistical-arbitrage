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

    def backtest(self, returns):
        train_idx, test_idx = split_train(returns.index.values, numInputs=self._train_window_size,
                                          numOutputs=self._split_window_size + 1, numJumps=1)
        train_idx = train_idx[::self._split_window_size]
        test_idx = test_idx[::self._split_window_size]

        # self._split_backtesters = []
        # strategy_returns = []
        # for idx_tr, idx_te in zip(train_idx, test_idx):
        #     backtester = SplitBacktester(preprocessor=copy.deepcopy(self._preprocessor),
        #                                  trader=copy.deepcopy(self._trader))
        #     backtester = backtester.fit(returns.loc[idx_tr])
        #     self._split_backtesters.append(backtester)
        #     backtester_returns = backtester.backtest(returns.loc[idx_te])
        #     strategy_returns.append(backtester_returns)

        executor = ThreadPoolExecutor()
        self._split_backtesters = list(
            executor.map(lambda idx: SplitBacktester(preprocessor=copy.deepcopy(self._preprocessor),
                                                     trader=copy.deepcopy(self._trader)).fit(returns.loc[idx]),
                         train_idx))

        strategy_returns = [backtester.backtest(returns.loc[idx])
                            for idx, backtester in zip(test_idx, self._split_backtesters)]
        return pd.concat(strategy_returns)

        # return pd.concat()
