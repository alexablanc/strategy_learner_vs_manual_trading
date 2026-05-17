import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import util as ut
import marketsimcode as msc
from ManualStrategy import ManualStrategy as ms
from StrategyLearner import StrategyLearner as sl

def get_benchmark(symbol, sd, ed):
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    prices_all.ffill(inplace=True)
    prices_all.bfill(inplace=True)
    trades = pd.DataFrame(0.0, index=prices_all.index, columns=[symbol])
    trades.iloc[0] = 1000
    return trades

def plot_experiment1(benchmark_norm, ms_norm, sl_norm, title, filename):
    plt.figure(figsize=(10, 6))

    benchmark_series = benchmark_norm.squeeze()
    ms_series = ms_norm.squeeze()
    sl_series = sl_norm.squeeze()

    plt.plot(benchmark_series.index, benchmark_series.values, color="purple", label="Benchmark")
    plt.plot(ms_series.index,ms_series.values,color="red",label="Manual Strategy")
    plt.plot(sl_series.index,sl_series.values,color="blue",label="Strategy Learner")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run():
    symbol = "JPM"
    sv = 100000
    commission = 9.95
    impact = 0.005

    #in sample period
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)

    # oos period
    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)

    #manual strategy
    manual_strat = ms(verbose=False, impact=impact, commission=commission)
    ms_trades_in = manual_strat.testPolicy(symbol=symbol, sd=sd_in,  ed=ed_in,  sv=sv)
    ms_trades_out = manual_strat.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)

    #Strategy Learner trained on in sample data only
    strat_learner = sl(verbose=False, impact=impact, commission=commission)
    strat_learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    sl_trades_in = strat_learner.testPolicy(symbol=symbol, sd=sd_in,  ed=ed_in,  sv=sv)
    sl_trades_out = strat_learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)

    #Benchmark
    bm_trades_in = get_benchmark(symbol, sd_in,  ed_in)
    bm_trades_out = get_benchmark(symbol, sd_out, ed_out)

    #portfolio values
    ms_portval_in = msc.compute_portvals(ms_trades_in,  sv=sv, commission=commission, impact=impact)
    ms_portval_out = msc.compute_portvals(ms_trades_out, sv=sv, commission=commission, impact=impact)
    sl_portval_in = msc.compute_portvals(sl_trades_in,  sv=sv, commission=commission, impact=impact)
    sl_portval_out = msc.compute_portvals(sl_trades_out, sv=sv, commission=commission, impact=impact)
    bm_portval_in = msc.compute_portvals(bm_trades_in,  sv=sv, commission=commission, impact=impact)
    bm_portval_out = msc.compute_portvals(bm_trades_out, sv=sv, commission=commission, impact=impact)

    # normalize
    ms_norm_in = ms_portval_in  / ms_portval_in.iloc[0]
    ms_norm_out = ms_portval_out / ms_portval_out.iloc[0]
    sl_norm_in = sl_portval_in  / sl_portval_in.iloc[0]
    sl_norm_out = sl_portval_out / sl_portval_out.iloc[0]
    bm_norm_in = bm_portval_in  / bm_portval_in.iloc[0]
    bm_norm_out = bm_portval_out / bm_portval_out.iloc[0]

    # chart 1 : in sample
    plot_experiment1(
        bm_norm_in, ms_norm_in, sl_norm_in,
        title=f"Manual Strategy vs Strategy Learner vs Benchmark — In-Sample ({symbol})",
        filename="experiment1_insample.png"
    )

    # chart 2 : oos
    plot_experiment1(
        bm_norm_out, ms_norm_out, sl_norm_out,
        title=f"Manual Strategy vs Strategy Learner vs Benchmark — Out-of-Sample ({symbol})",
        filename="experiment1_outsample.png"
    )