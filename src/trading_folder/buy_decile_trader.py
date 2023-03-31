from .buy_quantile_trader import BuyQuantileTrader

class BuyDecileTrader(BuyQuantileTrader):
    def __init__(self, d: int):
        super().__init__(10, d)


BuyDecileTradingRule = BuyDecileTrader
