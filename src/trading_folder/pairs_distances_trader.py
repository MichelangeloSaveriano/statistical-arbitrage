import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from .trader_base import TraderBase, TradingRule
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
    def __init__(self, p: int,
                 trading_rule: TradingRule | list[TradingRule] | dict[str, TradingRule] = None,
                 metric: str = 'correlation', use_returns: bool = True, weighting: str = None):
        # Number of pairs
        self._p = p

        self._metric = metric
        self._use_returns = use_returns
        self._pairs = None
        self._B = None

        self._trading_rule = trading_rule
        if self._trading_rule is None:
            self._trading_rule = StdTradingRule()
        elif isinstance(trading_rule, list):
            self._trading_rule = dict(zip(range(len(trading_rule)), trading_rule))

        self._weighting = weighting

    def train(self, X_train: pd.DataFrame,
              y_train: pd.DataFrame = None):
        n_assets = X_train.shape[1]
        data_matrix = X_train.values

        self._pairs = compute_minimum_distance_pairs(X_train, self._p,
                                                     self._metric,
                                                     self._use_returns)

        self._B = np.zeros((n_assets, self._p))
        self._B[self._pairs[0], np.arange(self._p)] = 1
        self._B[self._pairs[1], np.arange(self._p)] = -1

        pairs_distances = data_matrix @ self._B

        if isinstance(self._trading_rule, dict):
            for trader in self._trading_rule.values():
                trader.train(pairs_distances)
        else:
            self._trading_rule.train(pairs_distances)

        return self

    def compute_trading_mask(self, X_test: pd.DataFrame) -> pd.DataFrame | dict[pd.DataFrame]:
        pairs_distances = pd.DataFrame(X_test @ self._B)

        if isinstance(self._trading_rule, dict):
            trading_mask_dict = dict()
            for trading_rule_name, trading_rule in self._trading_rule.items():
                pairs_trading_mask = trading_rule.compute_trading_mask(pairs_distances)
                trading_mask = self.pairs_trading_mask_to_asset_trading_mask(pairs_trading_mask).fillna(0)
                trading_mask.index = X_test.index
                trading_mask.columns = X_test.columns
                trading_mask_dict[trading_rule_name] = trading_mask

            return trading_mask_dict

        pairs_trading_mask = self._trading_rule.compute_trading_mask(pairs_distances)
        trading_mask = pd.DataFrame(self.pairs_trading_mask_to_asset_trading_mask(pairs_trading_mask),
                                    index=X_test.index, columns=X_test.columns).fillna(0)

        return trading_mask

    def pairs_trading_mask_to_asset_trading_mask(self, pairs_trading_mask: pd.DataFrame) -> pd.DataFrame:
        trading_mask = pairs_trading_mask.values @ self._B.T

        short_mask = (trading_mask < 0).astype(float)
        long_mask = (trading_mask > 0).astype(float)

        if self._weighting == 'proportional':
            short_mask *= -trading_mask
            long_mask *= trading_mask

        short_mask /= np.fmax(1e-6, short_mask.sum(axis=1).reshape((-1, 1)))
        long_mask /= np.fmax(1e-6, long_mask.sum(axis=1).reshape((-1, 1)))

        trading_mask = long_mask - short_mask
        return pd.DataFrame(trading_mask)
