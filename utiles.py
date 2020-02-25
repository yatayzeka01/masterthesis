from pandas import read_csv, to_datetime, DataFrame
import os
from datetime import datetime
from numpy import array, random
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import pandas
from finta import TA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
from screeninfo import get_monitors

np.set_printoptions(threshold=np.inf)
pandas.set_option('display.max_rows', None)


def find_time_interval(input_path, output_path, min_time_interval_size, min_number_of_distinct_days, start_date_inp):
    if os.path.exists(output_path):
        os.remove(output_path)

    files = os.listdir(input_path)
    exec_sum_dict = {}
    date_list = []
    number_of_valid_stocks = 0
    valid_stocks_list = []

    for file in files:
        (stock, ext) = os.path.splitext(file)
        if ext != '.csv':
            continue
        fullPath = os.path.join(input_path, file)
        tempdata = read_csv(fullPath)
        tempdata = tempdata[tempdata.Open != 0.0]
        tempdata = tempdata[tempdata.Volume != 0]
        info = tempdata[['Date', 'Time']]
        hoursDict = {k: g["Time"].tolist() for k, g in info.groupby("Date")}
        number_of_distinct_days = info["Date"].nunique()

        start_date = datetime.strptime(info['Date'].iloc[0], '%d.%m.%Y').date()
        end_date = datetime.strptime(info['Date'].iloc[-1], '%d.%m.%Y').date()
        start_threshold = datetime.strptime(start_date_inp, '%d.%m.%Y').date()
        date_list.append(start_date)
        date_list.append(end_date)

        hoursInDay = []
        for value in hoursDict.values():
            if value not in hoursInDay and len(value) >= min_time_interval_size:
                hoursInDay.append(value)

        if len(hoursInDay) != 0:
            x = sorted(set.intersection(*map(set, hoursInDay)))
            if len(
                    x) >= min_time_interval_size and start_date > start_threshold and number_of_distinct_days > min_number_of_distinct_days:
                number_of_valid_stocks += 1
                valid_stocks_list.append(stock)
                write_to_file(output_path, 'Stock name {0}'.format(stock))

                key = repr(x)
                if key not in exec_sum_dict.keys():
                    exec_sum_dict[key] = 1
                else:
                    exec_sum_dict[key] += 1

                write_to_file(output_path, 'The most frequent time intervals for {0} stock: {1}'.format(stock, x))
                write_to_file(output_path, 'Start date for {0} stock: {1}'.format(stock, start_date))
                write_to_file(output_path, 'Last date for {0} stock: {1}'.format(stock, end_date))
                write_to_file(output_path, 'Number of distinct days in {0} stock data: {1}\n'.format(stock,
                                                                                                     number_of_distinct_days))

    write_to_file(output_path, 'Executive Summary')
    write_to_file(output_path, 'Number of stocks in the dataset: {0}'.format(len(files)))
    write_to_file(output_path, 'Number of valid stocks in the dataset: {0}'.format(number_of_valid_stocks))
    write_to_file(output_path, 'The Oldest date in the dataset: {0}'.format(sorted(date_list)[0]))
    write_to_file(output_path, 'The most recent date in the dataset: {0}'.format(sorted(date_list)[-1]))
    write_to_file(output_path, 'The most frequent time intervals for the dataset: {0}'.format(exec_sum_dict))

    time_interval = sorted(exec_sum_dict, key=exec_sum_dict.get, reverse=True)[0]

    return time_interval, valid_stocks_list


def write_to_file(filename, text):
    fh = open(filename, 'a')
    fh.write(text + "\n")
    fh.close()


def prepare(input_path, time_interval, valid_stocks_list, tiFlag):
    # leno is the length of most_frequent_time interval that is 7 for our case.
    leno = len(time_interval)
    files = os.listdir(input_path)

    # Create empty dataframes consists of following columns
    combinedDataNormal = DataFrame(columns=['date_time', 'Stock', 'Volume', 'Open'])
    combinedDataTi = DataFrame(columns=['date_time', 'Stock', 'Sma', 'Volume', 'Open'])
    # Read all files in the input_path and append them to combinedData dataframe accordingly.

    for file in files:
        (stock, ext) = os.path.splitext(file)
        if ext != '.csv':
            continue
        if stock not in valid_stocks_list:
            continue

        fullPath = os.path.join(input_path, file)
        # tempdata type is pandas dataframe
        tempdata = read_csv(fullPath)

        # Discard the rows with Volume and Open values as 0
        tempdata = tempdata[tempdata.Volume != 0]
        tempdata = tempdata[tempdata.Open != 0.0]
        # Create a dataframe using tempdata and acquire the rows within the given time_interval
        newData = tempdata.loc[tempdata['Time'].isin(time_interval)]
        # Make an aggregation on newData and discard the shortDays
        numTimes = newData.groupby('Date').count()
        shortDays = numTimes[numTimes.Open != leno].index.values
        newData = newData[~newData['Date'].isin(shortDays)]

        # Add stock and date_time columns to the data
        newData['Stock'] = stock
        newData['date_time'] = to_datetime(newData['date_time'], format="%d.%m.%Y %H:%M")

        # Append the newData to combinedData

        if tiFlag:

            newData = technical_indicators(newData)
            newData = newData.reindex(['date_time', 'Stock', 'Sma', 'Volume', 'Open'], axis=1)
            cols = ['date_time', 'Stock', 'Sma', 'Volume', 'Open']
            newDataOrdered = newData[cols]
            combinedDataTi = combinedDataTi.append(newDataOrdered)
        else:
            newData = newData.drop(['Time', 'Close', 'High', 'Low', 'Date'], axis=1)
            newDataOrdered = newData.reindex(['date_time', 'Stock', 'Volume', 'Open'], axis=1)
            combinedDataNormal = combinedDataNormal.append(newDataOrdered)

    if tiFlag:
        combinedData = combinedDataTi
    else:
        combinedData = combinedDataNormal

    # Drop the duplicates and data with NaN values
    combinedData = combinedData.drop_duplicates()
    combinedData = combinedData.reset_index(drop=True)
    combinedData = combinedData.dropna()

    # Scale the Volume and Open columns between 0 and 1
    combinedData = minmaxnormalize(combinedData, time_interval)

    # Write out short info about the dataset
    info(combinedData, time_interval, input_path)
    combinedDatatoFileDf = combinedData.copy()
    combinedDatatoPlot = combinedData.copy()
    combinedDatatoPlot.set_index('date_time', inplace=True)
    droplist = plotter(combinedDatatoPlot)
    combinedDataToFile = datetime_separator(combinedDatatoFileDf)
    combinedDataToFile = combinedDataToFile[~combinedDataToFile['Stock'].isin(droplist)]

    if os.path.exists('combinedData.csv'):
        os.remove('combinedData.csv')
    combinedDataToFile.to_csv('combinedData.csv', encoding='utf-8', index=False)

    # Turn the pandas dataframe combinedData into a numpy array called finalData
    combinedData = combinedData[~combinedData['Stock'].isin(droplist)]
    finalData = array(combinedData)
    # Reshape the finalData accordingly. e.g. (2917, 7, 4). 2917 days, 7 hours and 4 columns
    finalData = finalData.reshape((int(finalData.shape[0] / leno), leno, finalData.shape[1]))
    random.shuffle(finalData)

    print(finalData.shape)
    time.sleep(10)
    return finalData


def technical_indicators(tiDF):
    # tiDF 'date_time', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Stock'

    ohlcv = tiDF[['date_time', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    ohlcv.rename(
        columns={'date_time': 'Date', 'Date': 'day', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                 'Volume': 'volume'}, inplace=True)
    ohlcv.set_index('Date', inplace=True)
    ohlc = ohlcv[['open', 'high', 'low', 'close']]

    ohlcv['sma'] = TA.SMA(ohlcv, 2)
    stock = tiDF['Stock'].unique().tolist()[0]
    ohlcv['Stock'] = stock

    ohlcv = ohlcv.dropna()

    numTimes = ohlcv.groupby('day').count()
    shortDays = numTimes[numTimes.sma != 7].index.values
    ohlcv = ohlcv[~ohlcv['day'].isin(shortDays)]

    ohlcv = ohlcv.drop(['high', 'low', 'close', 'day'], axis=1)
    ohlcv.reset_index(level=0, inplace=True)
    ohlcv.rename(columns={'Date': 'date_time', 'open': 'Open', 'volume': 'Volume', 'sma': 'Sma'}, inplace=True)
    ohlcv = ohlcv.reindex(['date_time', 'Stock', 'Sma', 'Volume', 'Open'], axis=1)

    return ohlcv


def minmaxnormalize(dfToNormalize, time_interval):
    # Clean the data after normalization and remove the rows with Volume and Open values as 0.0
    leno = len(time_interval)
    mms = MinMaxScaler()
    dfToNormalize[['Volume', 'Open', 'Sma']] = mms.fit_transform(dfToNormalize[['Volume', 'Open', 'Sma']])

    zerodays = dfToNormalize.loc[
        (dfToNormalize['Open'] == 0.0) | (dfToNormalize['Volume'] == 0.0) | (dfToNormalize['Sma'] == 0.0)]
    zerodays['date_to_remove'] = zerodays['date_time'].dt.date
    zerodayextended = zerodays[['date_to_remove', 'Stock']]

    zerodayslistindex = zerodayextended['date_to_remove'].to_string(index=False).split('\n')
    zerodayslistvalue = zerodayextended['Stock'].to_string(index=False).split('\n')

    timestampindex = []
    timestampvalue = []
    for i in zerodayslistindex:
        for h in time_interval:
            time_extension = i + " " + h + ":00"
            timestampindex.append(time_extension)

    for vv in zerodayslistvalue:
        for _ in range(leno):
            timestampvalue.append(vv)

    timestampdict = dict(zip(timestampindex, timestampvalue))

    for k, v in timestampdict.items():
        dfToNormalizeSelected = dfToNormalize.loc[
            (dfToNormalize['date_time'] == str(k).strip()) & (dfToNormalize['Stock'] == str(v).strip())]
        dfToNormalize = dfToNormalize.drop(dfToNormalizeSelected.index)

    return dfToNormalize


def info(infoDF, time_interval, input_path):
    print("=" * 50)
    print("Input path of the Dataset ", "\n")
    print(input_path, "\n")

    # time_interval is ['10:00', '11:00', '12:00', '14:00', '15:00', '16:00', '17:00'] for our case
    print("=" * 50)
    print("The Most Frequent Time Interval of the Dataset ", "\n")
    print(time_interval, "\n")

    print("=" * 50)
    print("First Five Rows ", "\n")
    print(infoDF.head(5), "\n")

    print("=" * 50)
    print("Information About Dataset", "\n")
    print(infoDF.info(), "\n")

    print("=" * 50)
    print("Describe the Dataset ", "\n")
    print(infoDF.describe(), "\n")

    print("=" * 50)
    print("Null Values t ", "\n")
    print(infoDF.isnull().sum(), "\n")

    print("=" * 50)
    print("Unique Stocks of the Dataset: ", "\n")
    print(infoDF.Stock.unique(), "\n")
    print("Total Number of Unique Stocks: ", infoDF.Stock.nunique(), "\n")


def datetime_separator(df):
    # df['date_time'] = datetime.to_string(df['date_time'], format="%d.%m.%Y %H:%M")
    df['date_time'] = df['date_time'].dt.strftime("%d.%m.%Y %H:%M")
    df['date'] = df['date_time'].str.extract('(\d\d.\d\d.\d\d\d\d)', expand=True)
    df['time'] = df['date_time'].str.extract('(\d\d:\d\d)', expand=True)
    dfOrdered = df.reindex(['date_time', 'date', 'time', 'Stock', 'Sma', 'Volume', 'Open'], axis=1)
    return dfOrdered


def plotter(combinedDatatoPlot):
    for m in get_monitors():
        width = m.width
        height = m.height
    fig = plt.figure(figsize=(width / 100., height / 100.), dpi=100)

    thrashlist = []
    stocks = combinedDatatoPlot[['Stock', 'Open']]
    num_of_stocks = len(stocks.Stock.unique())
    color = iter(plt.cm.jet(np.linspace(0, 1, num_of_stocks)))
    for stock in stocks.Stock.unique():
        c = next(color)
        stockDFRaw = stocks[stocks.Stock == stock]
        stockDF = pandas.DataFrame({stock: stockDFRaw["Open"]})
        stockDF.reset_index(level=0, inplace=True)
        thresholdUp = stockDF[stockDF[stock] > 0.4]
        if not thresholdUp.empty:
            print("Open Price Above:", stock)
            thrashlist.append(stock)
            continue
        thresholdDown = stockDF[stockDF[stock] < 0.005]
        if not thresholdDown.empty:
            print("Open Price Below:", stock)
            thrashlist.append(stock)
            continue
        plotindex = random.randrange(1, len(stockDF.index) - 1)
        plt.plot(stockDF["date_time"], stockDF[stock], c=c, label=stock)
        plt.ylabel('Open Price')
        plt.xlabel('Date')
        plt.legend(loc="upper left", fontsize=10)
        plt.annotate(stock, (mdates.date2num(stockDF["date_time"][plotindex]), stockDF[stock][plotindex]), xytext=(15, 15),
                     textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color=c), color=c)
    fig.savefig("open_plot.png", dpi=600)

    plt.clf()
    fig = plt.figure(figsize=(width / 100., height / 100.), dpi=100)

    stocks = combinedDatatoPlot[['Stock', 'Volume']]
    num_of_stocks = len(stocks.Stock.unique())
    color = iter(plt.cm.jet(np.linspace(0, 1, num_of_stocks)))
    for stock in stocks.Stock.unique():
        if stock in thrashlist:
            continue
        c = next(color)
        stockDFRaw = stocks[stocks.Stock == stock]
        stockDF = pandas.DataFrame({stock: stockDFRaw["Volume"]})
        stockDF.reset_index(level=0, inplace=True)
        thresholdUp = stockDF[stockDF[stock] > 0.4]
        if not thresholdUp.empty:
            print("Volume Above:", stock)
            thrashlist.append(stock)
            continue
        thresholdDown = stockDF[stockDF[stock] < 0.0000005]
        if not thresholdDown.empty:
            print("Volume Below:", stock)
            thrashlist.append(stock)
            continue
        plotindex = random.randrange(1, len(stockDF.index) - 1)
        plt.plot(stockDF["date_time"], stockDF[stock], c=c, label=stock)
        plt.ylabel('Volume')
        plt.xlabel('Date')
        plt.legend(loc="upper left", fontsize=10)
        plt.annotate(stock, (mdates.date2num(stockDF["date_time"][plotindex]), stockDF[stock][plotindex]), xytext=(15, 15),
                     textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color=c), color=c)
    fig.savefig("volume_plot.png", dpi=600)
    plt.close()
    return thrashlist
