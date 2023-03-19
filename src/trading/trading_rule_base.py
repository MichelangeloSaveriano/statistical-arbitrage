class TradingRuleBase:
    def fit(self, train, test=None):
        pass

    def transform(self, test):
        pass

    def fit_transform(self, train):
        self.fit(train)
        return self.transform(train)
