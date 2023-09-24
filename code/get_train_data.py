import numpy as np
from pandas_datareader import data as pdr
import datetime as dt
from yahooquery import Ticker
import pandas as pd
# import pickle 


def get_training_data(stock_index: str, start: dt.datetime, end: dt.datetime) -> None:
    """
    Downloads and saves training data from Yahoo
    Inputs: 
        index: the underlying index you want to predict
        start: starting date of data collection
        end: end date of data collection
    Outputs:
        Saves stock and price data in a csv
    """
    df=pd.DataFrame()
    colnames=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    available_indices=['nasdaq', 'dax']
    new_symbols=[]
    try:
        symbols= pd.read_csv('ticker_lists/ticker_'+stock_index+'.csv', sep=',',names=['Symbol'], skiprows=1)
    except: 
        print("Available indexes are: ", available_indices)
        raise

    for i,symbol in enumerate(symbols["Symbol"]):
        if stock_index=="dax":
            symbol=symbol+".DE"
        t = Ticker(symbol, asynchronous=True)
        df_single = t.history(start=start, end=end, interval='1d')
        if i==0:
           df=df_single
           new_symbols.append(symbol)
        else:
           df=pd.concat([df, df_single], axis=0)
    df.to_csv('train_data/train_'+stock_index+".csv")


if __name__== "__main__":
    start=dt.datetime(year=2015,month=1, day=1)
    end=dt.datetime(year=2020, month=1, day=1)
    stock_index="nasdaq"
    get_training_data(stock_index=stock_index, start=start, end=end)


