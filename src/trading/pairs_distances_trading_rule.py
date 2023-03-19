import pandas as pd
import numpy as np
from trading_rule_base import TradingRuleBase


def compute_distance_matrix(data_matrix, use_correlations):
    df = data_matrix
    if not isinstance(data_matrix, pd.DataFrame):
        df = pd.DataFrame(data_matrix)
    if not use_correlations:  # use prices instead
        df = df.cumsum()
    corr = df.corr()
    return corr.values


class SpreadsTradingRule(TradingRuleBase):
    def __init__(self, p, use_correlations=True, gamma=2):
        self._p = p
        self._use_correlations = use_correlations
        self._pairs = None
        self._gamma = gamma
        self._thresholds = None
        pass

    def fit(self, train, test=None):
        data_matrix = train
        if isinstance(train, pd.DataFrame):
            data_matrix = train.values

        idx_row, idx_col = np.triu_indices(train.shape[1], k=1)
        distance_matrix = compute_distance_matrix(train, self._use_correlations)
        triu_vals = distance_matrix[idx_row, idx_col]
        minimum_distance_pairs_idx = np.argsort(triu_vals)[::-1][:self._p]
        self._pairs = (idx_row[minimum_distance_pairs_idx],
                       idx_col[minimum_distance_pairs_idx])

        distances = data_matrix[:, self._pairs[0]] - data_matrix[:, self._pairs[1]]
        self._thresholds = distances.std(axis=0, keepdims=True) * self._gamma

    def transform(self, test):
        data_matrix = test
        if isinstance(test, pd.DataFrame):
            data_matrix = test.values

        distances = data_matrix[:, self._pairs[0]] - data_matrix[:, self._pairs[1]]

        trigger_short = distances >= self._thresholds
        trigger_long = distances <= -self._thresholds

        short_mask = np.zeros_like(data_matrix)
        short_mask[:, self._pairs[0]] += trigger_short
        short_mask[:, self._pairs[1]] += trigger_long

        long_mask = np.zeros_like(data_matrix)
        long_mask[:, self._pairs[1]] += trigger_short
        long_mask[:, self._pairs[0]] += trigger_long

        short_mask /= np.fmax(1, short_mask.sum(axis=1).reshape((-1, 1)))
        long_mask /= np.fmax(1, long_mask.sum(axis=1).reshape((-1, 1)))

        trading_mask = long_mask - short_mask

        if isinstance(test, pd.DataFrame):
            trading_mask = pd.DataFrame(trading_mask,
                                        index=test.index, columns=test.columns)
            trading_mask.fillna(0)

        return trading_mask
