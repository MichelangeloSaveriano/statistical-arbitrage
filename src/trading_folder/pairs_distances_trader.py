import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from .trader_base import TraderBase
from .std_trader import StdTradingRule


def compute_distance_matrix(data_matrix, metric='correlation', use_returns=True):
    if not use_returns:  # use prices instead
        data_matrix = data_matrix.cumsum(axis=0)
    distance_matrix = pairwise_distances(data_matrix.T, metric=metric)
    return distance_matrix


def minimum_distance_pairs(distance_matrix, p):
    idx_row, idx_col = np.triu_indices(distance_matrix.shape[1], k=1)
    triu_vals = distance_matrix[idx_row, idx_col]
    # minimum_distance_pairs_idx = np.argsort(triu_vals)[::-1][:p]
    minimum_distance_pairs_idx = np.argsort(triu_vals)[:p]
    pairs = (idx_row[minimum_distance_pairs_idx],
             idx_col[minimum_distance_pairs_idx])
    return pairs


def compute_minimum_distance_pairs(X_train, p, metric, use_returns):
    distance_matrix = compute_distance_matrix(X_train,
                                              metric=metric,
                                              use_returns=use_returns)
    pairs = minimum_distance_pairs(distance_matrix, p)
    return pairs


class PairsDistanceTrader(TraderBase):
    def __init__(self, p, trading_rule=None, metric='correlation',
                 use_returns=True, weighting=None):
        # Number of pairs
        self._p = p

        self._metric = metric
        self._use_returns = use_returns
        self._pairs = None
        self._B = None

        self._trading_rule = trading_rule
        if self._trading_rule is None:
            self._trading_rule = StdTradingRule()
        self._weighting = weighting


    def train(self, X_train, y_train=None):
        n_assets = X_train.shape[1]
        data_matrix = X_train
        if isinstance(X_train, pd.DataFrame):
            data_matrix = X_train.values

        self._pairs = compute_minimum_distance_pairs(X_train, self._p,
                                                     self._metric,
                                                     self._use_returns)

        self._B = np.zeros((n_assets, self._p))
        self._B[self._pairs[0], np.arange(self._p)] = 1
        self._B[self._pairs[1], np.arange(self._p)] = -1

        pairs_distances = data_matrix @ self._B
        self._trading_rule.train(pairs_distances)

    def compute_trading_mask(self, X_test):
        data_matrix = X_test
        if isinstance(X_test, pd.DataFrame):
            data_matrix = X_test.values

        pairs_distances = data_matrix @ self._B
        pairs_trading_mask = self._trading_rule.compute_trading_mask(pairs_distances)
        trading_mask = pairs_trading_mask @ self._B.T

        short_mask = (trading_mask < 0).astype(float)
        long_mask = (trading_mask > 0).astype(float)

        if self._weighting == 'proportional':
            short_mask *= -trading_mask
            long_mask *= trading_mask

        short_mask /= np.fmax(1e-6, short_mask.sum(axis=1).reshape((-1, 1)))
        long_mask /= np.fmax(1e-6, long_mask.sum(axis=1).reshape((-1, 1)))

        trading_mask = long_mask - short_mask

        if isinstance(X_test, pd.DataFrame):
            trading_mask = pd.DataFrame(trading_mask,
                                        index=X_test.index, columns=X_test.columns)
            trading_mask.fillna(0)

        return trading_mask
