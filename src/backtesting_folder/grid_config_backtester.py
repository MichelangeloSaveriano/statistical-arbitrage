from typing import Dict

from .configuration_backtester import ConfigBacktester

import pandas as pd
from tqdm import tqdm

class GridBacktester:
    def __init__(self, configs: Dict[str, ConfigBacktester], verbose=False):
        self._configs = configs
        self._verbose = verbose

    def fit(self, returns):
        backtesters = self._configs.values()
        if self._verbose:
            backtesters = tqdm(self._configs.values())

        for backtester in backtesters:
            backtester.fit(returns)

    def backtest(self, returns):
        return pd.DataFrame({
            config_name: backtester.backtest(returns) for config_name, backtester in self._configs.items()
        })

    def fit_backtest(self, returns):
        self.fit(returns)
        return self.backtest(returns)
