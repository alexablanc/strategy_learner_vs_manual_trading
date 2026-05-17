import datetime as dt
import pandas as pd
from util import get_data
import indicators as ind


class ManualStrategy(object):
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000):

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = get_data(syms, dates)
        prices_all.ffill(inplace=True)
        prices_all.bfill(inplace=True)

        # pass series into ind function
        prices = prices_all[symbol]  # <-- single bracket = Series

        bb = ind.bollinger_percent_b(prices, window=20)
        mom = ind.momentum(prices, window=10)
        rsi = ind.rsi(prices, window=14)

        # same date range, drop nans
        combined = pd.DataFrame({
            "bb": bb,
            "mom": mom,
            "rsi": rsi
        }).dropna()

        signal = pd.Series(0, index=combined.index)

        buy_score = (
            (combined["bb"] < 0.2).astype(int) +
            (combined["mom"] < 0.0).astype(int) +
            (combined["rsi"] < 40).astype(int)
        )

        sell_score = (
            (combined["bb"] > 0.8).astype(int) +
            (combined["mom"] > 0.0).astype(int) +
            (combined["rsi"] > 60).astype(int)
        )
        signal[buy_score >= 2] = 1
        signal[sell_score >= 2] = -1

        trades = pd.DataFrame(0.0, index=prices_all.index, columns=[symbol])
        current_pos = 0

        for date in combined.index:
            desired_pos = signal[date] * 1000  # 1000, 0, or -1000
            if desired_pos != current_pos:
                order = desired_pos - current_pos  # handles 1000, -1000, 2000, -2000
                trades.loc[date, symbol] = order
                current_pos = desired_pos

        return trades
