import numpy as np
import pandas as pd  		  	   		   	 			  		 			 	 	 		 		 	
from util import get_data
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
def compute_portvals(  		  	   		   	 			  		 			 	 	 		 		 	
    df_trades,
    start_val=1000000,  		  	   		   	 			  		 			 	 	 		 		 	
    commission=0.00,
    impact=0.00,
    ):
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Computes the portfolio values.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param df_trades, a single column data frame, indexed by date, whose values represent trades for each trading day   		  	   		   	 			  		 			 	 	 		 		 	
    :type df_trades: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
    :param start_val: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
    :type start_val: int  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit), default is 0  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction, default is 0  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 			  		 			 	 	 		 		 	
    :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
    """
    #GET START AND END DATE
    df_trades.sort_index(inplace=True)
    start_date = df_trades.index[0]
    end_date = df_trades.index[-1]

    #GET PRICE DATA
    symbols = df_trades.columns.tolist()
    dates = pd.date_range(start_date,end_date).tolist()
    df_prices = get_data(symbols, dates, colname="Adj Close")[symbols]
    df_prices['Cash'] = 1
    dates = df_prices.index

    # ADD CASH COLUMN TO df_trades
    symbol = symbols[0]
    df_trades_copy = df_trades.copy(deep=True)
    df_trades_copy['has_trade'] = 0
    df_trades_copy.loc[df_trades_copy[symbol]!=0, 'has_trade'] = 1
    df_trades_copy['Cash'] = 0
    df_trades_copy['Cash'] = -df_prices[symbol]*df_trades_copy[symbol] - commission*df_trades_copy['has_trade'] \
                             - df_prices[symbol]*np.abs(df_trades_copy[symbol])*impact*df_trades_copy['has_trade']

    #HOLDINGS
    df_holdings =  df_trades_copy.copy()
    df_holdings.drop(['has_trade'], axis=1, inplace=True)
    df_holdings.loc[dates[0], 'Cash'] += start_val
    df_holdings = df_holdings.cumsum()

    #VALUE
    df_value = df_prices*df_holdings
    portvals = pd.DataFrame(df_value.sum(axis=1), index=df_value.index, columns=['Portfolio'])

    return portvals  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
def test_code(df_trades):
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # this is a helper function you can use to test your code  		  	   		   	 			  		 			 	 	 		 		 	
    # note that during autograding his function will not be called.  		  	   		   	 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		   	 			  		 			 	 	 		 		 	

    sv = 1000000  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Process orders  		  	   		   	 			  		 			 	 	 		 		 	
    portvals = compute_portvals(df_trades, start_val=sv,commission=0, impact=0)
    if isinstance(portvals, pd.DataFrame):  		  	   		   	 			  		 			 	 	 		 		 	
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		   	 			  		 			 	 	 		 		 	
    else:  		  	   		   	 			  		 			 	 	 		 		 	
        "warning, code did not return a DataFrame"  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Get portfolio stats
    df_trades.sort_index(inplace=True)
    start_date = df_trades.index[0]
    end_date = df_trades.index[-1]
    dates = pd.date_range(start_date,end_date).tolist()
    df = get_data(['$SPX'], dates, colname="Adj Close")
    df = df.drop(['SPY'],axis=1)
    df['Fund'] = portvals
    daily_returns = (df / df.shift(1)) - 1
    daily_returns = daily_returns[1:]

    #stats for portfolio
    avg_daily_ret = daily_returns['Fund'].mean()
    std_daily_ret = daily_returns['Fund'].std()
    sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret
    cum_ret = (df['Fund'][-1] / df['Fund'][0] - 1)

    #stats for SPX
    avg_daily_ret_SPY = daily_returns['$SPX'].mean()
    std_daily_ret_SPY = daily_returns['$SPX'].std()
    sharpe_ratio_SPY =np.sqrt(252)*avg_daily_ret_SPY/std_daily_ret_SPY
    cum_ret_SPY = (df['$SPX'][-1] / df['$SPX'][0] - 1)

    # Compare portfolio against $SPX  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Date Range: {start_date} to {end_date}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"# of Days: {portvals.shape[0]}")
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 			  		 			 	 	 		 		 	
    #print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    #print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    #print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    #print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Final Portfolio Value: {portvals[-1]}")

if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    test_code()  		  	   		   	 			  		 			 	 	 		 		 	
