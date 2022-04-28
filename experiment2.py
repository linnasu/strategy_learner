"""

Project 8: Strategy Evaluation - Experiment 2
Student Name: Linna Su
GT User ID: lsu63
GT ID: 903640548

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data
import util as ut
import random
import ManualStrategy as MS
import StrategyLearner as SL
import marketsimcode as sim

def comparePerf(df_combined, verbose = False):

	daily_returns = (df_combined / df_combined.shift(1)) - 1
	daily_returns = daily_returns[1:]

	df_normalized = df_combined / df_combined.iloc[0]
	xaxis = df_normalized.index
	pd.plotting.register_matplotlib_converters()
	plt.figure(figsize=(12, 8))
	plt.title('Normalized Portfolio Value for Strategy Learner with Different Market Impact - In Sample', fontsize=15)
	plt.plot(xaxis, df_normalized.iloc[:, 0], label="Impact = 0.0001")
	plt.plot(xaxis, df_normalized.iloc[:, 1], label="Impact = 0.0005")
	plt.plot(xaxis, df_normalized.iloc[:, 2], label="Impact = 0.005")
	plt.plot(xaxis, df_normalized.iloc[:, 3], label="Impact = 0.05")

	plt.xlabel('Date')
	plt.xticks(fontsize=12)
	plt.ylabel('Normalized Value ($)')
	plt.yticks(fontsize=12)
	plt.legend()
	plt.grid()
	plt.savefig("Experiment2.png")
	plt.clf()

	### Print Stats ###
	cum_ret1 = df_combined.iloc[:,0][-1] / df_combined.iloc[:,0][0] - 1
	avg_daily_ret1 = daily_returns.iloc[:,0].mean()
	std_daily_ret1 = daily_returns.iloc[:,0].std()

	cum_ret2 = df_combined.iloc[:, 1][-1] / df_combined.iloc[:, 1][0] - 1
	avg_daily_ret2 = daily_returns.iloc[:, 1].mean()
	std_daily_ret2 = daily_returns.iloc[:, 1].std()

	cum_ret3 = df_combined.iloc[:, 2][-1] / df_combined.iloc[:, 2][0] - 1
	avg_daily_ret3 = daily_returns.iloc[:, 2].mean()
	std_daily_ret3 = daily_returns.iloc[:, 2].std()

	cum_ret4 = df_combined.iloc[:, 3][-1] / df_combined.iloc[:, 3][0] - 1
	avg_daily_ret4 = daily_returns.iloc[:, 3].mean()
	std_daily_ret4 = daily_returns.iloc[:, 3].std()

	if verbose:
		print('-------------------------------------------------------------------------------------------')
		results = pd.DataFrame(0, index=['Cumulative return', 'Stdev of daily returns', 'Mean of daily returns'],
                               columns=['impact=0.0001', 'impact=0.0005','impact=0.005', 'impact=0.05'])
		results.loc['Cumulative return', :] = round(cum_ret1, 4), round(cum_ret2, 4), round(cum_ret3, 4), round(cum_ret4, 4)
		results.loc['Stdev of daily returns', :] = round(std_daily_ret1, 4), round(std_daily_ret2, 4), round(std_daily_ret3, 4), round(std_daily_ret4, 4)
		results.loc['Mean of daily returns', :] = round(avg_daily_ret1, 4), round(avg_daily_ret2, 4), round(avg_daily_ret3, 4), round(avg_daily_ret4, 4)
		print(results)
		print('-------------------------------------------------------------------------------------------')


def author():
	return 'lsu63'


if __name__ == "__main__":
	symbol = 'JPM'
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	sv = 100000
	commission = 0
	impact1 = 0.0001
	impact2 = 0.0005
	impact3 = 0.005
	impact4 = 0.05
	verbose = False

	# case 1
	learner1 = SL.StrategyLearner(verbose=False, impact=impact1, commission=commission)
	learner1.add_evidence(symbol, sd, ed)
	learner_trades1 = learner1.testPolicy(symbol, sd, ed)
	portval_learner1 = sim.compute_portvals(learner_trades1, sv, commission, impact1)

	# case 2
	learner2 = SL.StrategyLearner(verbose=False, impact=impact2, commission=commission)
	learner2.add_evidence(symbol, sd, ed)
	learner_trades2 = learner2.testPolicy(symbol, sd, ed)
	portval_learner2 = sim.compute_portvals(learner_trades2, sv, commission, impact2)\

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

	verbose = True
	comparePerf(df_combined, verbose)

