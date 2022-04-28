""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Linna Su   		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: lsu63  		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903640548 		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	

import numpy as np
import pandas as pd
import datetime as dt  		  	   		   	 			  		 			 	 	 		 		 	
import random
import util as ut
from indicators import EMA, BBP, MACD, Momentum, OBV
import RTLearner as rtl
import BagLearner as bl

  		  	   		   	 			  		 			 	 	 		 		 	
class StrategyLearner(object):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # constructor  		  	   		   	 			  		 			 	 	 		 		 	
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		   	 			  		 			 	 	 		 		 	
        self.impact = impact  		  	   		   	 			  		 			 	 	 		 		 	
        self.commission = commission
        self.learner = None

  		  	   		   	 			  		 			 	 	 		 		 	
    # this method should create a QLearner, and train it for trading  		  	   		   	 			  		 			 	 	 		 		 	
    def add_evidence(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="AAPL",
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 12/31/2009  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        # add your code to do learning here
        # construct data_x for training
        ema = np.array(EMA(symbol, sd, ed, 21))
        ema = (ema - ema.mean())/ema.std()
        bbp = np.array(BBP(symbol, sd, ed, 21))
        bbp = (bbp - bbp.mean())/bbp.std()
        macd = np.array(MACD(symbol, sd, ed))
        macd = (macd - macd.mean())/macd.std()
        momentum = np.array(Momentum(symbol, sd, ed, 21))
        momentum = (momentum - momentum.mean())/momentum.std()
        data_x = np.hstack((ema, bbp, macd, momentum))


        # construct data_y for training
        return_window = 10
        new_ed = pd.bdate_range(start=ed, periods=2*return_window, freq='B')[-1]
        dates = pd.date_range(sd, new_ed).tolist()
        prices = ut.get_data([symbol], dates, colname="Adj Close")
        prices.drop(['SPY'], axis=1, inplace=True)
        returns = prices.shift(-return_window)/prices - 1
        returns = returns.loc[sd:ed,:]
        YBUY = 2*(self.impact + self.commission/sv)
        YSELL = -YBUY

        data_y = pd.DataFrame(0,index=returns.index, columns=[symbol])
        data_y.loc[returns[symbol] > YBUY, symbol] = 1
        data_y.loc[returns[symbol] < YSELL, symbol] = -1
        data_y = np.array(data_y[symbol])

        #create learner#
        leaf_size = 10
        bags = 20
        self.learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={'leaf_size':leaf_size}, bags=bags, boost=False, verbose=False)
        self.learner.add_evidence(data_x, data_y)


  		  	   		   	 			  		 			 	 	 		 		 	
    # this method should use the existing policy and test it against new data  		  	   		   	 			  		 			 	 	 		 		 	
    def testPolicy(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="AAPL",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2010  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 12/31/2011  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	

        # construct data_x for testing
        ema = np.array(EMA(symbol, sd, ed, 21))
        ema = (ema - ema.mean())/ema.std()
        bbp = np.array(BBP(symbol, sd, ed, 21))
        bbp = (bbp - bbp.mean())/bbp.std()
        macd = np.array(MACD(symbol, sd, ed))
        macd = (macd - macd.mean())/macd.std()
        momentum = np.array(Momentum(symbol, sd, ed, 21))
        momentum = (momentum - momentum.mean())/momentum.std()
        data_x = np.hstack((ema, bbp, macd, momentum))

        # predict y
        pred_y_oos = self.learner.query(data_x)

        # derive holdings and trades based on predicted y
        dates = EMA(symbol, sd, ed, 21).index
        df_holdings = pd.DataFrame(pred_y_oos,index=dates, columns=[symbol])
        df_holdings.loc[df_holdings[symbol] == 1,symbol] = 1000
        df_holdings.loc[df_holdings[symbol] == -1, symbol] = -1000
        df_holdings.loc[df_holdings[symbol] == 0, symbol] = 0
        df_trades = df_holdings - df_holdings.shift(1)
        df_trades.iloc[0] = df_holdings.iloc[0]

        if self.verbose:  		  	   		   	 			  		 			 	 	 		 		 	
            print(df_trades)

        return df_trades
  		  	   		   	 			  		 			 	 	 		 		 	
def author():
    return 'lsu63'


if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    print("This is Strategy Learner")
