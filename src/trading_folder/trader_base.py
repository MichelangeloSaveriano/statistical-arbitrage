from abc import ABC, abstractmethod


class TraderBase(ABC):
    @abstractmethod
    def fit(self, train, test=None):
        pass

    @abstractmethod
    def transform(self, test):
        pass

    def fit_transform(self, train):
        self.fit(train)
        return self.transform(train)
