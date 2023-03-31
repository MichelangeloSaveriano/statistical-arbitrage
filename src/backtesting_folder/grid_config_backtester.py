from typing import Dict

from .configuration_backtester import ConfigBacktester

import pandas as pd
from tqdm import tqdm


class GridBacktester:
    def __init__(self, configs: Dict[str, ConfigBacktester], verbose=False):
        self._backtesters_dict = configs
        self._verbose = verbose

    def fit(self, returns):
        backtesters = self._backtesters_dict.values()
        if self._verbose:
            backtesters = tqdm(self._backtesters_dict.values())

        for backtester in backtesters:
            backtester.fit(returns)

    def backtest(self, returns):
        result_dataframes = dict()
        for config_name, backtester in self._backtesters_dict.items():
            result_dataframe = backtester.backtest(returns)
            if len(result_dataframe.columns) == 1 and result_dataframe.columns[0] == 0:
                result_dataframe = result_dataframe[0]
            result_dataframes[config_name] = result_dataframe

        # result_dataframes_dict = {
        #     config_name: backtester.backtest(returns)[0] for config_name, backtester in self._backtesters_dict.items()
        # }

        return pd.concat(result_dataframes.values(), axis=1, keys=result_dataframes.keys())
        # return pd.DataFrame(result_dataframes_dict)

    def fit_backtest(self, returns):
        self.fit(returns)
        return self.backtest(returns)

    @property
    def backtesters_dict(self):
        return self._backtesters_dict
