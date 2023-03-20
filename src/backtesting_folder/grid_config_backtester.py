from configuration_backtester import ConfigBacktester

import pandas as pd


class GridBacktester:
    def __init__(self, configs: dict[str, ConfigBacktester]):
        self._configs = configs

    def backtest(self, returns):
        return pd.DataFrame({
            config_name: backtester.backtest(returns) for config_name, backtester in self._configs.items()
        })
