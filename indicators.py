"""

Project 6: Indicator Evaluation
Student Name: Linna Su
GT User ID: lsu63
GT ID: 903640548

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
import util as ut

##Indicator 1: Price/Exponential Moving Average
def EMA(symbol, sd, ed, lookback, verbose=False):
	new_sd = pd.bdate_range(end=sd,periods = 2*lookback, freq='B')[0]
	dates = pd.date_range(new_sd, ed).tolist()
	prices = get_data([symbol], dates, colname="Adj Close")
	prices.drop(['SPY'], axis=1, inplace = True)
	ema = prices.ewm(span=lookback, adjust=False).mean().loc[sd:,:]
	price_over_ema = prices.loc[sd:,:]/ema

	if verbose:
		plt.figure(figsize=(12, 8))
		xaxis = ema.index
		plt.subplot(211)
		plt.plot(xaxis, prices.loc[sd:,:], label="price", color="k")
		plt.plot(xaxis, ema, label="{}-Day EMA".format(lookback), color="b")
		plt.title("{} Price and {}-Day EMA".format(symbol, lookback), fontsize=18)
		plt.ylabel("Price ($)")
		plt.grid()
		plt.legend()

		plt.subplot(212)
		plt.plot(xaxis, price_over_ema, label="price/{}-Day EMA".format(lookback), color="b")
		plt.hlines(y=1.2, xmin=xaxis[0], xmax=xaxis[-1], colors='r', linestyles='dashed')
		plt.hlines(y=0.7, xmin=xaxis[0], xmax=xaxis[-1], colors='r', linestyles='dashed')
		plt.title("Price / {}-Day EMA".format(lookback),fontsize=18)
		plt.xlabel("Date")
		plt.ylabel("Value")
		plt.grid()
		plt.legend()

		plt.savefig("Indicator1_ema.png")
		plt.clf()

	return price_over_ema


##Indicator 2: % Bollinger Bands
def BBP(symbol, sd, ed, lookback, verbose=False):
	new_sd = pd.bdate_range(end=sd, periods=2 * lookback, freq='B')[0]
	dates = pd.date_range(new_sd, ed).tolist()
	prices = get_data([symbol], dates, colname="Adj Close")
	prices.drop(['SPY'], axis=1, inplace = True)
	sma = prices.rolling(window = lookback, min_periods = lookback).mean().loc[sd:,:]
	rolling_std = prices.rolling(window = lookback, min_periods = lookback).std().loc[sd:,:]
	upper_band = sma + 2*rolling_std
	lower_band = sma - 2*rolling_std
	bbp = (prices.loc[sma.index,:] - lower_band)/(upper_band-lower_band)

	if verbose:
		plt.figure(figsize=(12, 8))
		xaxis = sma.index
		plt.subplot(211)
		plt.plot(xaxis, prices.loc[sma.index], label="Price", color="k")
		plt.plot(xaxis, sma, label="SMA", color="g")
		plt.plot(xaxis, upper_band, label="Upper Band", color="b")
		plt.plot(xaxis, lower_band, label="Lower Band", color="b")
		plt.title("{} Price, {}-Day SMA and Bollinger Bands".format(symbol,lookback), fontsize=18)
		plt.ylabel(ylabel="Price ($)")
		plt.grid()
		plt.legend()

		plt.subplot(212)
		plt.plot(xaxis, bbp, label="BBP Signal", color="b")
		plt.title("% B - Rolling {} Days".format(lookback), fontsize=18)
		plt.hlines(y=1, xmin=xaxis[0], xmax=xaxis[-1], colors='r', linestyles='dashed')
		plt.hlines(y=0, xmin=xaxis[0], xmax=xaxis[-1], colors='r', linestyles='dashed')
		plt.xlabel(xlabel="Date")
		plt.ylabel(ylabel="Value")
		plt.grid()
		plt.legend()
		plt.savefig('Indicator2_bbp.png')
		plt.clf()

	return bbp

##Indicator 3: Momentum
def Momentum(symbol, sd, ed, lookback, verbose = False):
	new_sd = pd.bdate_range(end=sd, periods=2 * lookback, freq='B')[0]
	dates = pd.date_range(new_sd, ed).tolist()
	prices = get_data([symbol], dates, colname="Adj Close")
	prices.drop(['SPY'], axis=1, inplace=True)
	momentum = (prices/prices.shift(lookback)-1).loc[sd:,:]
	prices = prices.loc[sd:,:]
	prices = prices/prices.iloc[0]
	centerline = momentum.mean()
	if verbose:
		plt.figure(figsize=(12, 8))
		xaxis = momentum.index

		plt.plot(xaxis, prices, label="Price", color ="k")
		plt.plot(xaxis, momentum, label="{}Day Momentum".format(lookback), color = "b")
		plt.hlines(y=centerline, xmin=xaxis[0], xmax=xaxis[-1], colors='r', linestyles='dashed')
		plt.title("{} Price Normalized and {}Day Momentum".format(symbol, lookback), fontsize=18)
		plt.xlabel(xlabel="Date")
		plt.ylabel(ylabel="Normalized Price ($) /Value")
		plt.grid()
		plt.legend()
		plt.savefig('Indicator3_Momentum.png')

	return momentum

##Indicator 4: MACD - MACD Signal
def MACD(symbol, sd, ed, lookbacks=[12,26,9], verbose=False):
	new_sd = pd.bdate_range(end=sd, periods=2 * lookbacks[1], freq='B')[0]
	dates = pd.date_range(new_sd, ed).tolist()
	prices = get_data([symbol], dates, colname="Adj Close")
	prices.drop(['SPY'], axis=1, inplace=True)
	ema_fast = prices.ewm(span=lookbacks[0], adjust=False).mean()
	ema_slow = prices.ewm(span=lookbacks[1], adjust=False).mean()
	macd = ema_fast - ema_slow
	macd_signal = macd.ewm(span=lookbacks[2], adjust=False).mean().loc[sd:,:]
	macd = macd.loc[sd:,:]
	indicator = macd-macd_signal

	if verbose:
		plt.figure(figsize=(12, 8))
		xaxis = macd.index
		plt.subplot(211)
		plt.plot(xaxis, prices.loc[sd:, :], label="price", color="k")
		plt.title("{} Price".format(symbol), fontsize=18)
		plt.ylabel("Price ($)")
		plt.grid()
		plt.legend()

		plt.subplot(212)
		plt.plot(xaxis, macd, label="MACD ({}day-{}day)".format(lookbacks[0],lookbacks[1]), color="b")
		plt.plot(xaxis, macd_signal, label="MACD Signal ({}-Day EMA)".format(lookbacks[2]), color="g")
		plt.plot(xaxis, indicator, label="MACD - MACD Signal", color="r")
		plt.title("MACD, MACD Signal and MACD - MACD Signal", fontsize=18)
		plt.xlabel("Date")
		plt.ylabel("Value")
		plt.grid()
		plt.legend()

		plt.savefig("Indicator4_macd.png")
		plt.clf()

	return indicator


##Indicator 5: On Balance Volume Change
def OBV(symbol, sd, ed, lookback, verbose=False):
	new_sd = pd.bdate_range(end=sd, periods=2 * lookback, freq='B')[0]
	dates = pd.date_range(new_sd, ed).tolist()
	prices = get_data([symbol], dates, colname="Adj Close")
	prices.drop(['SPY'], axis=1, inplace=True)
	volume = get_data([symbol], dates, colname="Volume")
	volume.drop(['SPY'], axis=1, inplace=True)
	obv = (np.sign(prices.diff()) * volume).dropna().cumsum()
	obv_change = (obv - obv.shift(lookback)).loc[sd:,:]
	obv = obv.loc[sd:,:]
	momentum = (prices / prices.shift(lookback) - 1).loc[sd:, :]
	prices = prices.loc[sd:,:]
	prices = prices/prices.iloc[0]

	if verbose:
		plt.figure(figsize=(12, 8))
		xaxis = obv.index
		plt.subplot(211)
		plt.plot(xaxis, prices.loc[sd:, :], label="price", color="k")
		plt.plot(xaxis, momentum, label="{}Day Momentum".format(lookback), color="b")
		plt.title("{} Price Normalized and {}Day Momentum".format(symbol, lookback), fontsize=18)
		plt.ylabel("Normalized Price ($) /Value")
		plt.grid()
		plt.legend()

		plt.subplot(212)
		plt.plot(xaxis, obv, label="OBV", color="b")
		plt.plot(xaxis, obv_change, label="{}Day OBV Change".format(lookback), color="r")

		plt.title("OBV and OBV Change")
		plt.xlabel("Date")
		plt.ylabel("Volume")
		plt.grid()
		plt.legend()

		plt.savefig("Indicator5_obv.png")
		plt.clf()


	return obv_change


def author():
	return 'lsu63'