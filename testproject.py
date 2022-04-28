import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
import datetime as dt
import util as ut
import random
import ManualStrategy as MS
import StrategyLearner as SL
import marketsimcode as sim
import experiment1 as ex1
import experiment2 as ex2
import argparse

def benchMark(symbol, sd, ed):
    dates = pd.date_range(sd, ed).tolist()
    df_prices = get_data([symbol], dates, colname="Adj Close")
    df_prices.drop(['SPY'], axis=1, inplace=True)
    df_trades = pd.DataFrame(0, index=df_prices.index, columns=[symbol])
    df_trades.iloc[0] = 1000

    return df_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--print2screen', default=False, action='store_true')
    args = parser.parse_args()
    print_to_screen = args.print2screen

    symbol = 'JPM'
    sd_train = dt.datetime(2008, 1, 1)
    ed_train = dt.datetime(2009, 12, 31)
    sd_test = dt.datetime(2010, 1, 1)
    ed_test = dt.datetime(2011, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005
    verbose = False
    ### Run manual strategy, generate charts and stats ###
    MS.comparePerf(symbol,sv,commission,impact,print_to_screen)

    ### Experiment 1 ###
    manual_trades = MS.testPolicy(symbol, sd_train, ed_train)
    learner = SL.StrategyLearner(verbose=False, impact=impact, commission=commission)
    random.seed(903640548)
    learner.add_evidence(symbol, sd_train, ed_train)
    learner_trades = learner.testPolicy(symbol, sd_train, ed_train)
    benchmark_trades = benchMark(symbol, sd_train, ed_train)
    ex1.comparePerf(manual_trades, learner_trades, benchmark_trades, sv, commission, impact,print_to_screen)

    ### Experiment 2 ###
    # case 1
    sd = sd_train
    ed = ed_train
    commission = 0
    impact1 = 0.0001
    impact2 = 0.0005
    impact3 = 0.005
    impact4 = 0.05
    # case 2
    learner1 = SL.StrategyLearner(verbose=False, impact=impact1, commission=commission)
    learner1.add_evidence(symbol, sd, ed)
    learner_trades1 = learner1.testPolicy(symbol, sd, ed)
    portval_learner1 = sim.compute_portvals(learner_trades1, sv, commission, impact1)

    # case 2
    learner2 = SL.StrategyLearner(verbose=False, impact=impact2, commission=commission)
    learner2.add_evidence(symbol, sd, ed)
    learner_trades2 = learner2.testPolicy(symbol, sd, ed)
    portval_learner2 = sim.compute_portvals(learner_trades2, sv, commission, impact2)

    # case 3
    learner3 = SL.StrategyLearner(verbose=False, impact=impact3, commission=commission)
    learner3.add_evidence(symbol, sd, ed)
    learner_trades3 = learner3.testPolicy(symbol, sd, ed)
    portval_learner3 = sim.compute_portvals(learner_trades3, sv, commission, impact3)

    # case 4
    learner4 = SL.StrategyLearner(verbose=False, impact=impact4, commission=commission)
    learner4.add_evidence(symbol, sd, ed)
    learner_trades4 = learner4.testPolicy(symbol, sd, ed)
    portval_learner4 = sim.compute_portvals(learner_trades4, sv, commission, impact4)

    df_combined = pd.concat([portval_learner1, portval_learner2, portval_learner3, portval_learner4], axis=1)

    ex2.comparePerf(df_combined, print_to_screen)





