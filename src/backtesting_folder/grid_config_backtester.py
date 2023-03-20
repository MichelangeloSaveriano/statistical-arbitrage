from typing import Dict

from .configuration_backtester import ConfigBacktester

import pandas as pd
from tqdm import tqdm

class GridBacktester:
    def __init__(self, configs: Dict[str, ConfigBacktester], verbose=False):
        self._configs = configs
        self._verbose = verbose

    def backtest(self, returns):
        items = self._configs.items()
        if self._verbose:
            items = tqdm(self._configs.items())

        return pd.DataFrame({
            config_name: backtester.backtest(returns) for config_name, backtester in items
        })
