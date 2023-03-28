from src.backtesting import *
from src.trading import *
from src.laplacian_estimators import *
from src.preprocessing import *

import pandas as pd
import numpy as np

# Load Data
prices = pd.read_csv("./data/sp500_prices_long.csv", parse_dates=['Date'])
prices['month'] = prices['Date'].dt.month
prices['year'] = prices['Date'].dt.year
raw_monthly_prices = prices.groupby(['year', 'month']).last().reset_index()
raw_monthly_prices['year_month'] = raw_monthly_prices['year'].astype(str) + ('00' + raw_monthly_prices['month'].astype(str)).str[-2:]
raw_monthly_prices = raw_monthly_prices.set_index('year_month').drop(columns=['year', 'month', 'Date'])
monthly_prices = raw_monthly_prices.loc[:, raw_monthly_prices.isna().mean(axis=0) < .5]
monthly_log_returns = np.log(monthly_prices).diff().iloc[1:]
# print(monthly_log_returns)

# Load Factors
factors = pd.read_csv("./data/F-F_Research_Data_5_Factors_2x3.csv", skiprows=2)
factors = factors[factors['Date'].astype(str).str.strip().str.len() == 6]
factors = factors.set_index('Date')
factors = factors.astype(np.float64)

# Merge Indexes
idx = monthly_log_returns.merge(factors, right_index=True, left_index=True).index
monthly_log_returns = monthly_log_returns.loc[idx]
factors = factors.loc[idx]

# Remove Risk-free Returns
monthly_log_returns_rf = monthly_log_returns - factors[['RF']].values / 100
factors_rf = factors.drop(columns='RF')


X_train = monthly_log_returns_rf.dropna()# .sample(250, axis=1)
print(X_train)
X_train_1 = X_train.values[:60]
X_train_2 = X_train.values[12:72]
X_train_3 = X_train.values[120:]

from src.laplacian_estimators_folder.graph_learning_algorithm import learn_connected_graph_heavy_tails
from scipy.spatial.distance import cosine

print('\n--- X Train 1 ---\n')
results1 = learn_connected_graph_heavy_tails(X_train_1, heavy_type='gaussian', verbose=True, max_iter=500, tau=3)
w1 = results1['w']
print('\n--- X Train 2 ---\n')
results2 = learn_connected_graph_heavy_tails(X_train_2, w0=w1, heavy_type='gaussian', verbose=True, max_iter=500,
                                             rho=1, tau=3)
w2 = results2['w']
print('\n--- X Train 2 - No w0 ---\n')
print(cosine(w1, w2))
print(w1-w2)

results2 = learn_connected_graph_heavy_tails(X_train_2, heavy_type='gaussian', verbose=True, max_iter=500)
print('Fine')


# Backtest Strategies
# config_backtesters = dict()
# config_backtesters['NoPreprocessing Identity'] = ConfigBacktester(
#     preprocessor=NoPreprocessing(),
#     trader=SpreadsTrader(laplacian_estimator=LaplacianIdentityLaplacianEstimator())
# )
# config_backtesters['NoPreprocessing Corr-50'] = ConfigBacktester(
#     preprocessor=NoPreprocessing(),
#     trader=SpreadsTrader(laplacian_estimator=LaplacianCorrKLaplacianEstimator(k=50))
# )
# config_backtesters['Residuals Identity'] = ConfigBacktester(
#     preprocessor=ResidualsPreprocessing(factors_rf),
#     trader=SpreadsTrader(laplacian_estimator=LaplacianIdentityLaplacianEstimator())
# )
# config_backtesters['Residuals Corr-50'] = ConfigBacktester(
#     preprocessor=ResidualsPreprocessing(factors_rf),
#     trader=SpreadsTrader(laplacian_estimator=LaplacianCorrKLaplacianEstimator(k=50)),
# )

# config_backtesters['NoPreprocessing Pairs-20'] = ConfigBacktester(
#     preprocessor=NoPreprocessing(),
#     trader=PairsDistanceTrader(20)
# )
#
# config_backtesters['Residuals Pairs-20'] = ConfigBacktester(
#     preprocessor=ResidualsPreprocessing(factors_rf),
#     trader=PairsDistanceTrader(20)
# )
# config_backtesters['NoPreprocessing LGMRF'] = ConfigBacktester(
#     preprocessor=NoPreprocessing(),
#     trader=SpreadsTradingRule(laplacian_estimator=LaplacianGraphLearningLaplacianEstimator()),
#     split_window_size=36
# )

# backtester = GridBacktester(config_backtesters, verbose=True)
#
# config_returns = backtester.backtest(monthly_log_returns)
# print(config_returns)
# # print(config_returns.corr())
# print((np.exp(np.log(1+config_returns).mean() * 12) - 1) * 100)

