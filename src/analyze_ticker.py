import pandas as pd
import matplotlib.pyplot as plt
import os

def full_analysis(ticker, data_dir="data/price_data"):
    """
    Run the full analysis pipeline for a given ticker, including price plotting,
    returns analysis, and moving average visualization.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol to analyze.
    data_dir : str, optional
        Directory where price CSV files are stored (default "data/price_data").

    Returns:
    --------
    None
        Displays plots for price, returns, histogram, and moving averages.
    """

    df = load_price_data(ticker, data_dir)
    df = calculate_daily_returns(df)

    plot_prices(df, ticker)
    plot_returns(df, ticker)
    plot_returns_histogram(df, ticker)

    df = calculate_moving_averages(df)
    plot_prices_with_ma(df, ticker)


def load_price_data(ticker, data_dir="data/price_data"):
    """
    Load CSV price data for a given ticker.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol.
    data_dir : str, optional
        Directory containing price CSV files (default "data/price_data").

    Returns:
    --------
    pandas.DataFrame
        DataFrame indexed by date containing price data including 'Close' prices.
    """
    path = os.path.join(data_dir, f"{ticker}_daily.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def calculate_daily_returns(df):
    """
    Calculate daily returns from closing prices and add as a new column.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data with a 'Close' column.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with an additional 'Daily Return' column representing daily percentage returns.
    """
    df['Daily Return'] = df['Close'].pct_change()
    return df


def plot_prices(df, ticker):
    """
    Plot closing prices over time for the given ticker.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data indexed by date and a 'Close' column.
    ticker : str
        Stock ticker symbol.

    Returns:
    --------
    None
        Displays a line plot of closing prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()


def plot_returns(df, ticker):
    """
    Plot daily returns over time for the given ticker.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a 'Daily Return' column.
    ticker : str
        Stock ticker symbol.

    Returns:
    --------
    None
        Displays a line plot of daily returns.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Daily Return'], label=f'{ticker} Daily Returns')
    plt.title(f'{ticker} Daily Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.show()


def plot_returns_histogram(df, ticker):
    """
    Plot a histogram of daily returns for the given ticker.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a 'Daily Return' column.
    ticker : str
        Stock ticker symbol.

    Returns:
    --------
    None
        Displays a histogram of daily returns.
    """
    plt.figure(figsize=(10, 5))
    df['Daily Return'].hist(bins=50)
    plt.title(f'{ticker} Daily Returns Histogram')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.show()


def calculate_moving_averages(df):
    """
    Calculate moving averages of closing prices for windows of 7, 30, 100, and 200 days.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Close' price data.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added columns: 'MA7', 'MA30', 'MA100', 'MA200' representing moving averages.
    """
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df


def plot_prices_with_ma(df, ticker):
    """
    Plot closing prices with moving averages for the given ticker.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Close' and moving average columns ('MA7', 'MA30', 'MA100', 'MA200').
    ticker : str
        Stock ticker symbol.

    Returns:
    --------
    None
        Displays a plot of closing prices overlaid with moving averages.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.7)
    plt.plot(df.index, df['MA7'], label='MA 7')
    plt.plot(df.index, df['MA30'], label='MA 30')
    plt.plot(df.index, df['MA100'], label='MA 100')
    plt.plot(df.index, df['MA200'], label='MA 200')
    plt.title(f'{ticker} Closing Prices with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()