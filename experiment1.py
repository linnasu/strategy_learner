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



def comparePerf(df_manual, df_learner, df_benchmark, sv = 100000, commission = 9.95, impact=0.005, verbose=False):
	"""
	    Computes the portfolio values and compare performance.

	    :param df_manual, a list of single column data frames, indexed by date, whose values represent trades for each trading day generated by manual strategy
	    :type data: pandas.DataFrame
	    :param df_learner, a list of single column data frames, indexed by date, whose values represent trades for each trading day generated by strategy learner
	    :type data: pandas.DataFrame
	    :param df_benchmark, a list of single column data frames, indexed by date, whose values represent trades for each trading day generated by benchmark strategy
	    :type data: pandas.DataFrame
	    :param sd: A datetime object that represents the start date
        :type sd: datetime
        :param ed: A datetime object that represents the end date
        :type ed: datetime
	    :param start_val: The starting value of the portfolio
	    :type start_val: int
	    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit), default is 0
	    :type commission: float
	    :param impact: The amount the price moves against the trader compared to the historical data at each transaction, default is 0
	    :type impact: float
	    :param verbose: If “verbose” is True, will print graph to the screen.
        				If verbose = False code will not print anything to the screen
        				Default will be False.
    	:type verbose: bool
	    """

	portval_manual = sim.compute_portvals(df_manual, sv, commission, impact)
	portval_learner = sim.compute_portvals(df_learner, sv, commission, impact)
	portval_benchmark = sim.compute_portvals(df_benchmark, sv, commission, impact)

	df_combined = pd.concat([portval_manual, portval_learner, portval_benchmark], axis=1)

	daily_returns = (df_combined / df_combined.shift(1)) - 1
	daily_returns = daily_returns[1:]

	df_normalized = df_combined / df_combined.iloc[0]
	xaxis = df_normalized.index
	pd.plotting.register_matplotlib_converters()
	plt.figure(figsize=(12, 8))
	plt.title('Normalized Portfolio Value for Manual Strategy, Strategy Learner and Benchmark - In Sample', fontsize=15)
	plt.plot(xaxis, df_normalized.iloc[:,0], label="Manual Strategy", color ="r")
	plt.plot(xaxis, df_normalized.iloc[:,1], label="Strategy Learner", color ="b")
	plt.plot(xaxis, df_normalized.iloc[:,2], label="Benchmark", color ="g")

	plt.xlabel('Date')
	plt.xticks(fontsize=12)
	plt.ylabel('Normalized Value ($)')
	plt.yticks(fontsize=12)
	plt.legend()
	plt.grid()
	plt.savefig("Experiment1.png")
	plt.clf()

	### Print Stats ###
	if verbose:
		cum_ret1 = df_combined.iloc[:, 0][-1] / df_combined.iloc[:, 0][0] - 1
		avg_daily_ret1 = daily_returns.iloc[:, 0].mean()
		std_daily_ret1 = daily_returns.iloc[:, 0].std()

		cum_ret2 = df_combined.iloc[:, 1][-1] / df_combined.iloc[:, 1][0] - 1
		avg_daily_ret2 = daily_returns.iloc[:, 1].mean()
		std_daily_ret2 = daily_returns.iloc[:, 1].std()

		cum_ret3 = df_combined.iloc[:, 2][-1] / df_combined.iloc[:, 2][0] - 1
		avg_daily_ret3 = daily_returns.iloc[:, 2].mean()
		std_daily_ret3 = daily_returns.iloc[:, 2].std()

		print('-------------------------------------------------------------------------------------------')
		results = pd.DataFrame(0, index=['Cumulative return', 'Stdev of daily returns', 'Mean of daily returns'],
							   columns=['Manual Strategy', 'Strategy Learner', 'Benchmark'])
		results.loc['Cumulative return', :] = round(cum_ret1, 4), round(cum_ret2, 4), round(cum_ret3, 4)
		results.loc['Stdev of daily returns', :] = round(std_daily_ret1, 4), round(std_daily_ret2, 4), round(std_daily_ret3, 4)
		results.loc['Mean of daily returns', :] = round(avg_daily_ret1, 4), round(avg_daily_ret2, 4), round(avg_daily_ret3, 4)
		print(results)
		print('-------------------------------------------------------------------------------------------')


def benchMark(symbol, sd, ed):
	dates = pd.date_range(sd, ed).tolist()
	df_prices = get_data([symbol], dates, colname="Adj Close")
	df_prices.drop(['SPY'], axis=1, inplace=True)
	df_trades = pd.DataFrame(0, index=df_prices.index, columns=[symbol])
	df_trades.iloc[0] = 1000

	return df_trades

if __name__ == "__main__":
	symbol = 'JPM'
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	sv = 100000
	commission = 9.95
	impact = 0.005
	verbose = False
	manual_trades = MS.testPolicy(symbol, sd, ed)
	learner = SL.StrategyLearner(verbose=False, impact=impact, commission=commission)
	learner.add_evidence(symbol, sd, ed)
	learner_trades = learner.testPolicy(symbol, sd, ed)
	benchmark_trades = benchMark(symbol, sd, ed)
	comparePerf(manual_trades,learner_trades, benchmark_trades, sv, commission, impact, verbose)
