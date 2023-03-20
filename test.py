from src.backtesting import *
from src.trading import *
from src.laplacian_estimators import *
from src.preprocessing import *

import pandas as pd
import numpy as np

config_backtesters = dict()
config_backtesters['NoPreprocessing Identity'] = ConfigBacktester(
    preprocessor=NoPreprocessing(),
    trader=SpreadsTradingRule(laplacian_estimator=LaplacianIdentityLaplacianEstimator())
)

config_backtesters['NoPreprocessing Corr-50'] = ConfigBacktester(
    preprocessor=NoPreprocessing(),
    trader=SpreadsTradingRule(laplacian_estimator=LaplacianCorrKLaplacianEstimator(k=50))
)

config_backtesters['NoPreprocessing LGMRF'] = ConfigBacktester(
    preprocessor=NoPreprocessing(),
    trader=SpreadsTradingRule(laplacian_estimator=LaplacianGraphLearningLaplacianEstimator()),
    split_window_size=36
)

backtester = GridBacktester(config_backtesters, verbose=True)

prices = pd.read_csv("data/sp500_prices_long.csv", parse_dates=['Date'])
prices['month'] = prices['Date'].dt.month
prices['year'] = prices['Date'].dt.year
raw_monthly_prices = prices.groupby(['year', 'month']).last().reset_index()
raw_monthly_prices['year_month'] = raw_monthly_prices['year'].astype(str) + ('00' + raw_monthly_prices['month'].astype(str)).str[-2:]
raw_monthly_prices = raw_monthly_prices.set_index('year_month').drop(columns=['year', 'month', 'Date'])
monthly_prices = raw_monthly_prices.loc[:, raw_monthly_prices.isna().mean(axis=0) < .5]
monthly_log_returns = np.log(monthly_prices).diff().iloc[1:]
print(monthly_log_returns)

config_returns = backtester.backtest(monthly_log_returns)
print(config_returns)
print(np.exp(np.mean(config_returns) * 12))

