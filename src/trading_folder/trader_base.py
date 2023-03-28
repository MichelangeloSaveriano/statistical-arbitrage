from abc import ABC, abstractmethod


class TraderBase(ABC):
    @abstractmethod
    def train(self, X_train, y_train=None):
        pass

    @abstractmethod
    def compute_trading_mask(self, X_test):
        pass
