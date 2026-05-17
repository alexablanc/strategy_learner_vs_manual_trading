import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import marketsimcode as msc
from StrategyLearner import StrategyLearner

def compute_num_trades(trades, symbol):
#return number of days where trades were non 0
    return int((trades[symbol] != 0).sum())

def compute_cum_return(portvals):
    pv = portvals.squeeze()
    return (pv.iloc[-1] / pv.iloc[0]) - 1

def save_table_figure(df, filename):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1), facecolor="white")
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.4, 1.6)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def run():
    symbol= "JPM"
    sv= 100000
    commission = 0.0

    # in sample period
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    impacts = [0.0, 0.002, 0.005, 0.010, 0.025, 0.050]

    cum_returns = []
    num_trades = []

    for impact in impacts:
        #train a new strategy learner at each impact level
        sl = StrategyLearner(verbose=False, impact=impact, commission=commission)
        sl.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        trades = sl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

        portvals = msc.compute_portvals(trades, sv=sv, commission=commission, impact=impact)
        cum_ret = compute_cum_return(portvals)
        n_trades = compute_num_trades(trades, symbol)

        cum_returns.append(cum_ret)
        num_trades.append(n_trades)

    # cum ret vs impact
    plt.figure(figsize=(8, 5))
    plt.plot(impacts, cum_returns, color="blue", marker="o", linewidth=2)
    plt.title(f"Experiment 2: Cumulative Return vs Impact ({symbol} In-Sample)")
    plt.xlabel("Impact")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment2_cumreturn.png")
    plt.close()

    # number of trades vs impact
    plt.figure(figsize=(8, 5))
    plt.plot(impacts, num_trades, color="red", marker="o", linewidth=2)
    plt.title(f"Experiment 2: Number of Trades vs Impact ({symbol} In-Sample)")
    plt.xlabel("Impact")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment2_numtrades.png")
    plt.close()

    # summary table
    results_df = pd.DataFrame({
        "Impact":           [f"{v:.3f}" for v in impacts],
        "Cumulative Return": [f"{v:.6f}" for v in cum_returns],
        "Number of Trades":  [str(v)     for v in num_trades],
    }).set_index("Impact")

    save_table_figure(results_df, "experiment2_table.png")