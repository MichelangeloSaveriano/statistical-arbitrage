from abc import ABC, abstractmethod


class TradingRuleBase(ABC):
    @abstractmethod
    def fit(self, train, test=None):
        pass

    @abstractmethod
    def transform(self, test):
        pass
