import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util as ut
import datetime as dt
from indicators import EMA, BBP, MACD, Momentum, OBV
import marketsimcode as sim

def testPolicy(symbol = "AAPL", sd = dt.datetime(2010, 1, 1), ed = dt.datetime(2011, 12, 31), sv = 100000):
	ema = EMA(symbol, sd, ed, 21)
	bbp = BBP(symbol, sd, ed, 21)
	macd = MACD(symbol, sd, ed)
	momentum = Momentum(symbol, sd, ed, 21)
	obv = OBV(symbol, sd, ed, 21)

	df = pd.DataFrame(0, index=bbp.index, columns=['feature_ema', 'feature_bbp', 'feature_macd', 'feature_momentum','feature_obv',
												   'signal_ema', 'signal_bbp','signal_macd','signal_momentum','signal_obv'])
	df.loc[:,'feature_ema'] = np.array(ema)
	df.loc[:,'feature_bbp'] = np.array(bbp)
	df.loc[:,'feature_macd'] = np.array(macd)
	df.loc[:,'feature_momentum'] = np.array(momentum)
	df.loc[:,'feature_obv'] = np.array(obv)
	df['signal_ema'].loc[df['feature_ema'] < 0.9] = 1
	df['signal_ema'].loc[df['feature_ema'] > 1.1] = -1
	df['signal_bbp'].loc[df['feature_bbp'] < 0] = 1
	df['signal_bbp'].loc[df['feature_bbp'] > 1] = -1
	df['feature_macd_crossover'] = np.sign(macd).diff()
	df['feature_macd_crossover'].iloc[0] = 0
	df['signal_macd'].loc[df['feature_macd_crossover'] > 0] = 1
	df['signal_macd'].loc[df['feature_macd_crossover'] < 0] = -1
	df['signal_momentum'].loc[df['feature_momentum'] <-0.15] = 1
	df['signal_momentum'].loc[df['feature_momentum'] > 0.15] = -1
	df['signal_obv'].loc[df['feature_obv'] > 0] = 1
	df['signal_obv'].loc[df['feature_obv'] < 0] = -1


	df['signal_combined'] = df['signal_ema'] + df['signal_bbp']  + df['signal_macd'] + df['signal_momentum'] # + df['signal_obv']
	df['output'] = 0
	df['output'].loc[df['signal_combined'] >= 1] = 1
	df['output'].loc[df['signal_combined'] <= -1] = -1

	df_holdings = pd.DataFrame(np.nan, index=df.index, columns=[symbol])
	df_holdings.loc[df['output'] > 0, symbol] = 1000
	df_holdings.loc[df['output'] < 0, symbol] = -1000
	df_holdings.ffill(inplace=True)
	df_holdings.fillna(0, inplace=True)

	df_trades = df_holdings - df_holdings.shift(1)
	df_trades.iloc[0] = df_holdings.iloc[0]

	return df_trades

def benchMark(symbol, sd, ed):
	dates = pd.date_range(sd, ed).tolist()
	df_prices = ut.get_data([symbol], dates, colname="Adj Close")
	df_prices.drop(['SPY'], axis=1, inplace=True)
	df_trades = pd.DataFrame(0, index=df_prices.index, columns=[symbol])
	df_trades.iloc[0] = 1000

	return df_trades

def comparePerf(symbol, sv = 100000, commission = 9.95, impact=0.05, verbose=False):
	sd_train = dt.datetime(2008, 1, 1)
	ed_train = dt.datetime(2009, 12, 31)
	sd_test = dt.datetime(2010, 1, 1)
	ed_test = dt.datetime(2011, 12, 31)

	#### Calculate portfolio Value####
	# in sample
	strategy_trades_is = testPolicy(symbol, sd_train, ed_train)
	benchmark_trades_is = benchMark(symbol, sd_train, ed_train)
	port_value_is = sim.compute_portvals(strategy_trades_is, sv, commission,  impact)
	port_value_is.columns = ['Manual Strategy_IS']
	benchmark_value_is = sim.compute_portvals(benchmark_trades_is, sv, commission,  impact)
	benchmark_value_is.columns = ['Benchmark_IS']
	df_combined_is = pd.concat([port_value_is,benchmark_value_is], axis = 1)
	daily_returns_is = (df_combined_is / df_combined_is.shift(1)) - 1
	daily_returns_is = daily_returns_is[1:]
	# out of sample
	strategy_trades_oos = testPolicy(symbol, sd_test, ed_test)
	benchmark_trades_oos = benchMark(symbol, sd_test, ed_test)
	port_value_oos = sim.compute_portvals(strategy_trades_oos, sv, commission,  impact)
	port_value_oos.columns = ['Manual Strategy_OOS']
	benchmark_value_oos = sim.compute_portvals(benchmark_trades_oos, sv, commission,  impact)
	benchmark_value_oos.columns = ['Benchmark_OOS']
	df_combined_oos = pd.concat([port_value_oos,benchmark_value_oos], axis = 1)
	daily_returns_oos = (df_combined_oos / df_combined_oos.shift(1)) - 1
	daily_returns_oos = daily_returns_oos[1:]

	###stats for Manual Strategy###
	# in sample
	cum_ret_is = (df_combined_is['Manual Strategy_IS'][-1] / df_combined_is['Manual Strategy_IS'][0] - 1)
	avg_daily_ret_is = daily_returns_is['Manual Strategy_IS'].mean()
	std_daily_ret_is = daily_returns_is['Manual Strategy_IS'].std()
	sharpe_ratio_is = np.sqrt(252) * avg_daily_ret_is / std_daily_ret_is
	# out of sample
	cum_ret_oos = (df_combined_oos['Manual Strategy_OOS'][-1] / df_combined_oos['Manual Strategy_OOS'][0] - 1)
	avg_daily_ret_oos = daily_returns_oos['Manual Strategy_OOS'].mean()
	std_daily_ret_oos = daily_returns_oos['Manual Strategy_OOS'].std()
	sharpe_ratio_oos = np.sqrt(252) * avg_daily_ret_oos / std_daily_ret_oos

	###stats for Benchmark###
	# in sample
	cum_ret_BM_is = (df_combined_is['Benchmark_IS'][-1] / df_combined_is['Benchmark_IS'][0] - 1)
	avg_daily_ret_BM_is = daily_returns_is['Benchmark_IS'].mean()
	std_daily_ret_BM_is = daily_returns_is['Benchmark_IS'].std()
	sharpe_ratio_BM_is =np.sqrt(252)*avg_daily_ret_BM_is/std_daily_ret_BM_is
	# out of sample
	cum_ret_BM_oos = (df_combined_oos['Benchmark_OOS'][-1] / df_combined_oos['Benchmark_OOS'][0] - 1)
	avg_daily_ret_BM_oos = daily_returns_oos['Benchmark_OOS'].mean()
	std_daily_ret_BM_oos = daily_returns_oos['Benchmark_OOS'].std()
	sharpe_ratio_BM_oos = np.sqrt(252) * avg_daily_ret_BM_oos / std_daily_ret_BM_oos


	### Plot ###
	# in sample
	df_normalized_is = df_combined_is/df_combined_is.iloc[0]
	xaxis = df_normalized_is.index
	pd.plotting.register_matplotlib_converters()
	plt.figure(figsize=(12, 8))
	plt.plot(xaxis, df_normalized_is[['Manual Strategy_IS']], label="Manual Strategy", color ="red")
	plt.plot(xaxis, df_normalized_is[['Benchmark_IS']], label="Benchmark", color="green")

	for i in strategy_trades_is[strategy_trades_is[symbol] > 0].index:
		plt.axvline(x=i, color='blue')
	for i in strategy_trades_is[strategy_trades_is[symbol] < 0].index:
		plt.axvline(x=i, color='black')

	plt.title('Normalized Portfolio Value for Manual Strategy and Benchmark Trading JPM - In Sample', fontsize=16)
	plt.xlabel('Date')
	plt.xticks(fontsize=12)
	plt.ylabel('Normalized Value ($)')
	plt.yticks(fontsize=12)
	plt.legend()
	plt.grid()
	plt.savefig("Manual Strategy vs_Benchmark - In Sample.png")
	plt.clf()

	#Plot out of sample
	df_normalized_oos = df_combined_oos/df_combined_oos.iloc[0]
	xaxis = df_normalized_oos.index
	plt.figure(figsize=(12, 8))
	plt.plot(xaxis, df_normalized_oos[['Manual Strategy_OOS']], label="Manual Strategy", color ="red")
	plt.plot(xaxis, df_normalized_oos[['Benchmark_OOS']], label="Benchmark", color="green")

	for i in strategy_trades_oos[strategy_trades_oos[symbol] > 0].index:
		plt.axvline(x=i, color='blue')
	for i in strategy_trades_oos[strategy_trades_oos[symbol] < 0].index:
		plt.axvline(x=i, color='black')

	plt.title('Normalized Portfolio Value for Manual Strategy and Benchmark Trading JPM - Out of Sample', fontsize=16)
	plt.xlabel('Date')
	plt.xticks(fontsize=12)
	plt.ylabel('Normalized Value ($)')
	plt.yticks(fontsize=12)
	plt.legend()
	plt.grid()
	plt.savefig("Manual Strategy vs_Benchmark - Out of Sample.png")
	plt.clf()

	### Print Stats ###

	if verbose:
		print('-------------------------------------------------------------------------------------------')
		results = pd.DataFrame(0, index=['Cumulative return', 'Stdev of daily returns', 'Mean of daily returns'],
							   columns=['Manual Strategy_IS', 'Benchmark_IS','Manual Strategy_OOS', 'Benchmark_OOS'])
		results.loc['Cumulative return', :] = round(cum_ret_is, 4), round(cum_ret_BM_is, 4), round(cum_ret_oos, 4), round(cum_ret_BM_oos, 4)
		results.loc['Stdev of daily returns', :] = round(std_daily_ret_is, 4), round(std_daily_ret_BM_is, 4), round(std_daily_ret_oos, 4), round(std_daily_ret_BM_oos, 4)
		results.loc['Mean of daily returns', :] = round(avg_daily_ret_is, 4), round(avg_daily_ret_BM_is, 4), round(avg_daily_ret_oos, 4), round(avg_daily_ret_BM_oos, 4)
		print(results)
		print('-------------------------------------------------------------------------------------------')



if __name__ == "__main__":
	print("This is Manual Strategy.")
	symbol = 'JPM'
	comparePerf(symbol, sv = 100000, commission = 9.95, impact=0.005, verbose=True)

