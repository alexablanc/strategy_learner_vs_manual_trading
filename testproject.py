import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util as ut
import marketsimcode as msc
from ManualStrategy import ManualStrategy as ms
from StrategyLearner import StrategyLearner as sl
import experiment1
import experiment2

np.random.seed(450000)

symbol= "JPM"
sv= 100000
commission= 9.95
impact= 0.005

sd_in= dt.datetime(2008, 1, 1)
ed_in= dt.datetime(2009, 12, 31)
sd_out= dt.datetime(2010, 1, 1)
ed_out= dt.datetime(2011, 12, 31)

#Manual Strategy
manual_strat= ms(verbose=False, impact=impact, commission=commission)
ms_trades_in= manual_strat.testPolicy(symbol=symbol, sd=sd_in,  ed=ed_in,  sv=sv)
ms_trades_out= manual_strat.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)

#Strategy Learner (train on in-sample only)
strat_learner= sl(verbose=False, impact=impact, commission=commission)
strat_learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
sl_trades_in= strat_learner.testPolicy(symbol=symbol, sd=sd_in,  ed=ed_in,  sv=sv)
sl_trades_out= strat_learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)

#Benchmark: buy 1000 shares on first trading day and hold
def get_benchmark(symbol, sd, ed):
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    prices_all.ffill(inplace=True)
    prices_all.bfill(inplace=True)
    trades = pd.DataFrame(0.0, index=prices_all.index, columns=[symbol])
    trades.iloc[0] = 1000
    return trades

bm_trades_in  = get_benchmark(symbol, sd_in,  ed_in)
bm_trades_out = get_benchmark(symbol, sd_out, ed_out)

#Portfolio values
ms_portval_in  = msc.compute_portvals(ms_trades_in,  sv=sv, commission=commission, impact=impact)
ms_portval_out = msc.compute_portvals(ms_trades_out, sv=sv, commission=commission, impact=impact)
sl_portval_in  = msc.compute_portvals(sl_trades_in,  sv=sv, commission=commission, impact=impact)
sl_portval_out = msc.compute_portvals(sl_trades_out, sv=sv, commission=commission, impact=impact)
bm_portval_in  = msc.compute_portvals(bm_trades_in,  sv=sv, commission=commission, impact=impact)
bm_portval_out = msc.compute_portvals(bm_trades_out, sv=sv, commission=commission, impact=impact)

#Normalize portvals
ms_norm_in  = ms_portval_in  / ms_portval_in.iloc[0]
ms_norm_out = ms_portval_out / ms_portval_out.iloc[0]
sl_norm_in  = sl_portval_in  / sl_portval_in.iloc[0]
sl_norm_out = sl_portval_out / sl_portval_out.iloc[0]
bm_norm_in  = bm_portval_in  / bm_portval_in.iloc[0]
bm_norm_out = bm_portval_out / bm_portval_out.iloc[0]

def add_entry_lines(ax, trades, symbol):
    position = 0
    for date, row in trades.iterrows():
        trade = row[symbol]
        if trade == 0:
            continue
        position += trade
        if position > 0:
            # long entry
            ax.axvline(x=date, color="blue",  linewidth=0.8, alpha=0.6)
        elif position < 0:
            # short entry
            ax.axvline(x=date, color="black", linewidth=0.8, alpha=0.6)

def plot_strategy_vs_benchmark(benchmark_norm, strategy_norm, title, filename,
                                trades=None, symbol=None):
    plt.figure(figsize=(10, 6))

    benchmark_series = benchmark_norm.squeeze()
    strategy_series  = strategy_norm.squeeze()

    plt.plot(benchmark_series.index, benchmark_series.values, color="purple", label="Benchmark")
    plt.plot(strategy_series.index,  strategy_series.values,  color="red",    label="Manual Strategy")

    if trades is not None and symbol is not None:
        ax = plt.gca()
        add_entry_lines(ax, trades, symbol)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_stats(portvals):
    #portvals: Series or one-column DataFrame of portfolio values
    #returns: cumulative return, std daily return, mean daily return, sharpe ratio

    portvals = portvals.squeeze()

    daily_ret      = portvals.pct_change().dropna()
    cum_ret        = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    std_daily_ret  = daily_ret.std()
    mean_daily_ret = daily_ret.mean()
    sharpe_ratio   = np.sqrt(252) * (mean_daily_ret / std_daily_ret) if std_daily_ret != 0 else 0.0

    return cum_ret, std_daily_ret, mean_daily_ret, sharpe_ratio

def save_table_figure(stats_df, filename="stats_table.png"):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    table = ax.table(
        cellText=stats_df.values,
        rowLabels=stats_df.index,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


#Charts
plot_strategy_vs_benchmark(
    bm_norm_in, ms_norm_in,
    title=f"Manual Strategy vs Benchmark — In-Sample ({symbol})",
    filename="manual_insample.png",
    trades=ms_trades_in, symbol=symbol
)

plot_strategy_vs_benchmark(
    bm_norm_out, ms_norm_out,
    title=f"Manual Strategy vs Benchmark — Out-of-Sample ({symbol})",
    filename="manual_outsample.png",
    trades=ms_trades_out, symbol=symbol
)

#In-Sample Stats Table
bm_in_cum,  bm_in_std,  bm_in_mean,  bm_in_sharpe  = compute_stats(bm_portval_in)
ms_in_cum,  ms_in_std,  ms_in_mean,  ms_in_sharpe  = compute_stats(ms_portval_in)
sl_in_cum,  sl_in_std,  sl_in_mean,  sl_in_sharpe  = compute_stats(sl_portval_in)

stats_in_df = pd.DataFrame({
    "Benchmark":        [bm_in_cum,  bm_in_std,  bm_in_mean,  bm_in_sharpe],
    "Manual Strategy":  [ms_in_cum,  ms_in_std,  ms_in_mean,  ms_in_sharpe],
    "Strategy Learner": [sl_in_cum,  sl_in_std,  sl_in_mean,  sl_in_sharpe],
}, index=[
    "Cumulative Return",
    "Std Dev of Daily Return",
    "Mean of Daily Return",
    "Sharpe Ratio",
])

# Format to 6 digits to the right of the decimal
stats_in_df = stats_in_df.applymap(lambda x: f"{x:.6f}")
save_table_figure(stats_in_df, "stats_table_insample.png")

#Out-of-Sample Stats Table
bm_out_cum, bm_out_std, bm_out_mean, bm_out_sharpe = compute_stats(bm_portval_out)
ms_out_cum, ms_out_std, ms_out_mean, ms_out_sharpe = compute_stats(ms_portval_out)
sl_out_cum, sl_out_std, sl_out_mean, sl_out_sharpe = compute_stats(sl_portval_out)

stats_out_df = pd.DataFrame({
    "Benchmark":        [bm_out_cum, bm_out_std, bm_out_mean, bm_out_sharpe],
    "Manual Strategy":  [ms_out_cum, ms_out_std, ms_out_mean, ms_out_sharpe],
    "Strategy Learner": [sl_out_cum, sl_out_std, sl_out_mean, sl_out_sharpe],
}, index=[
    "Cumulative Return",
    "Std Dev of Daily Return",
    "Mean of Daily Return",
    "Sharpe Ratio",
])

stats_out_df = stats_out_df.applymap(lambda x: f"{x:.6f}")
save_table_figure(stats_out_df, "stats_table_outsample.png")

experiment1.run()
experiment2.run()