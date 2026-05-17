import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

from util import get_data

def get_price_series(symbol="JPM",
                     sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2009, 12, 31)):

    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    prices = prices_all[[symbol]].copy()
    return prices[symbol]

def normalize(series):
    return series / series.iloc[0]

# Indicator functions

def bollinger_percent_b(prices, window=20):
    """
    Bollinger Bands:
    % = (Price - Lower Band) / (Upper Band - Lower Band)
    """
    sma = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = sma + 2 * rolling_std
    lower_band = sma - 2 * rolling_std

    percent_b = (prices - lower_band) / (upper_band - lower_band)
    percent_b.name = "Bollinger_%B"
    return percent_b


def rsi(prices, window=14):
    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi_vals = 100 - (100 / (1 + rs))
    rsi_vals.name = "RSI"
    return rsi_vals


def momentum(prices, window=10):
    """
    Momentum:
    momentum[t] = price[t] / price[t-window] - 1
    """
    mom = prices / prices.shift(window) - 1
    mom.name = "Momentum"
    return mom


def macd_histogram(prices, fast=12, slow=26, signal=9):
    """
    MACD Histogram as a SINGLE vector:
    MACD histogram = MACD line - signal line
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    hist = macd_line - signal_line
    hist.name = "MACD_Histogram"
    return hist


def price_ema_ratio(prices, window=20):
    ema = prices.ewm(span=window, adjust=False).mean()
    ratio = prices / ema
    ratio.name = "Price_EMA_Ratio"
    return ratio

# One figure per indicator

def plot_bollinger_percent_b(prices, window=20, filename="indicator_bollinger_percent_b.png"):
    sma = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = sma + 2 * rolling_std
    lower_band = sma - 2 * rolling_std
    percent_b = bollinger_percent_b(prices, window=window)
    base = prices.iloc[0]

    norm_price = prices / base
    norm_sma = sma / base
    norm_upper = upper_band / base
    norm_lower = lower_band / base

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(norm_price.index, norm_price, label="Normalized Price", color="blue")
    axes[0].plot(norm_sma.index, norm_sma, label=f"SMA ({window})", color="orange")
    axes[0].plot(norm_upper.index, norm_upper, label="Upper Band", color="green")
    axes[0].plot(norm_lower.index, norm_lower, label="Lower Band", color="red")
    axes[0].set_title("Bollinger Bands / %B")
    axes[0].set_ylabel("Normalized Price")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(percent_b.index, percent_b, label="%B")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].axhline(0.5, linestyle="--")
    axes[1].axhline(1.0, linestyle="--")
    axes[1].set_ylabel("%B")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_rsi(prices, window=14, filename="indicator_rsi.png"):
    rsi_vals = rsi(prices, window=window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(normalize(prices).index, normalize(prices), label="Normalized Price")
    axes[0].set_title("RSI")
    axes[0].set_ylabel("Normalized Price")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(rsi_vals.index, rsi_vals, label=f"RSI ({window})")
    axes[1].axhline(70, linestyle="--")
    axes[1].axhline(30, linestyle="--")
    axes[1].set_ylabel("RSI")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_momentum(prices, window=10, filename="indicator_momentum.png"):
    mom = momentum(prices, window=window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(normalize(prices).index, normalize(prices), label="Normalized Price")
    axes[0].set_title("Momentum")
    axes[0].set_ylabel("Normalized Price")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(mom.index, mom, label=f"Momentum ({window})")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_ylabel("Momentum")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_macd_histogram(prices, fast=12, slow=26, signal=9, filename="indicator_macd_histogram.png"):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_histogram(prices, fast=fast, slow=slow, signal=signal)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(normalize(prices).index, normalize(prices), label="Normalized Price")
    axes[0].set_title("MACD Histogram")
    axes[0].set_ylabel("Normalized Price")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(macd_line.index, macd_line, label="MACD Line")
    axes[1].plot(signal_line.index, signal_line, label="Signal Line")
    axes[1].bar(hist.index, hist, label="Histogram", width=3)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_ylabel("MACD")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_price_ema_ratio(prices, window=20, filename="indicator_price_ema_ratio.png"):
    ema = prices.ewm(span=window, adjust=False).mean()
    ratio = price_ema_ratio(prices, window=window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(normalize(prices).index, normalize(prices), label="Normalized Price")
    axes[0].plot(normalize(ema).index, normalize(ema), label=f"EMA ({window})")
    axes[0].set_title("Price / EMA Ratio")
    axes[0].set_ylabel("Normalized Price")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(ratio.index, ratio, label="Price / EMA")
    axes[1].axhline(1.0, linestyle="--")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Run all indicators and generate all plots

def run(symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31)):
    prices = get_price_series(symbol=symbol, sd=sd, ed=ed)

    plot_bollinger_percent_b(prices)
    plot_rsi(prices)
    plot_momentum(prices)
    plot_macd_histogram(prices)
    plot_price_ema_ratio(prices)

if __name__ == "__main__":
    run()