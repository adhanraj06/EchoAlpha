import pandas as pd
import matplotlib.pyplot as plt
import os


def full_analysis(ticker, out_dir, data_dir="data/price_data"):
    """
    Run the full analysis pipeline for a given ticker, including price plotting,
    returns analysis, and moving average visualization.

    Args:
        ticker (str): Ticker symbol of the stock to analyze.
        out_dir (str): Directory to save analysis outputs.
        data_dir (str, optional): Directory containing price CSV files.

    Returns:
        None
    """

    path = os.path.join(data_dir, f"{ticker}_daily.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df["Daily Return"] = df["Close"].pct_change()

    # Plot closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label=f"{ticker} Close Price")
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{ticker}_closing_prices.png"))
    plt.close()

    # Plot returns
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Daily Return"], label=f"{ticker} Daily Returns")
    plt.title(f"{ticker} Daily Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{ticker}_daily_returns.png"))
    plt.close()

    # Plot returns histogram
    plt.figure(figsize=(10, 5))
    df["Daily Return"].hist(bins=50)
    plt.title(f"{ticker} Daily Returns Histogram")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(out_dir, f"{ticker}_returns_histogram.png"))
    plt.close()

    # Calculate moving averages and plot prices with moving averages
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["MA100"] = df["Close"].rolling(window=100).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Close"], label="Close Price", alpha=0.7)
    plt.plot(df.index, df["MA7"], label="MA 7")
    plt.plot(df.index, df["MA30"], label="MA 30")
    plt.plot(df.index, df["MA100"], label="MA 100")
    plt.plot(df.index, df["MA200"], label="MA 200")
    plt.title(f"{ticker} Closing Prices with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{ticker}_moving_averages.png"))
    plt.close()
