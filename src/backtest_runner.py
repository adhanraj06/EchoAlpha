import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def load_price_data(ticker):
    """
    Load and format daily price data for a given ticker from CSV.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol.

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame indexed by date containing price data with columns such as 'Close' and 'Return'.
        Returns None if the data file is not found.
    """
    try:
        price_df = pd.read_csv(f"data/price_data/{ticker}_daily.csv", parse_dates=["Date"])
        price_df.set_index("Date", inplace=True)
        price_df.sort_index(inplace=True)
        return price_df
    except FileNotFoundError:
        print(f"⚠Missing Price Data for {ticker}")
        return None

def generate_signals_for_ticker(
    bull_threshold, bear_threshold, ticker, ticker_df, price_df, return_col,
    wait_weeks, hold_weeks, initial_balance=100000, transaction_cost=0.001
):
    """
    Generate trade signals and calculate returns for a single ticker based on sentiment and price reactions.

    Parameters:
    -----------
    bull_threshold : float
        Threshold ratio of positive to negative sentiment to classify as bullish.
    bear_threshold : float
        Threshold ratio of positive to negative sentiment to classify as bearish.
    ticker : str
        Stock ticker symbol.
    ticker_df : pandas.DataFrame
        DataFrame containing sentiment and return data for the ticker.
    price_df : pandas.DataFrame
        Daily price DataFrame indexed by date.
    return_col : str
        Column name in ticker_df representing the returns to use for signal generation.
    wait_weeks : int
        Number of weeks to wait after filing date before entering the trade.
    hold_weeks : int
        Number of weeks to hold the trade.
    initial_balance : float, optional
        Starting capital for trade simulations (default 100,000).
    transaction_cost : float, optional
        Proportional transaction cost per trade side (default 0.001).

    Returns:
    --------
    list of dict
        List of trade signal dictionaries containing trade metadata and performance metrics.
    """
    # define bullish and bearish sentiment measures (customizable)
    def is_bullish(row):
        if row["neg"] == 0:
            return row["pos"] > 0
        return (row["pos"] / row["neg"]) > bull_threshold

    def is_bearish(row):
        if row["neg"] == 0:
            return False
        return (row["pos"] / row["neg"]) < bear_threshold

    signals = []
    balance = initial_balance  # track running balance through trades

    for _, row in ticker_df.iterrows():
        filing_date = row["filing_date"]
        reaction = row[return_col]

        # convert to timestamp
        try:
            filing_date = pd.to_datetime(filing_date)
        except:
            continue

        # set signal direction
        sentiment_bull = is_bullish(row)
        sentiment_bear = is_bearish(row)

        if pd.isna(reaction):
            continue

        # determine signal type based on sentiment and price reaction
        signal = None
        if reaction < 0 and sentiment_bull:
            signal = "long"
        elif reaction > 0 and sentiment_bear:
            signal = "short"
        if not signal:
            continue

        # define entry and exit dates
        entry_date = filing_date + pd.Timedelta(weeks=wait_weeks)
        exit_date = entry_date + pd.Timedelta(weeks=hold_weeks)

        if entry_date not in price_df.index or exit_date not in price_df.index:
            continue

        entry_price = price_df.loc[entry_date]["Close"]
        exit_price = price_df.loc[exit_date]["Close"]

        ret = (exit_price - entry_price) / entry_price * 100
        if signal == "short":
            ret *= -1

        # calculate transaction costs for entry and exit (2x)
        cost = balance * transaction_cost * 2
        # calculate PnL net of transaction costs in dollars
        trade_pnl = (ret / 100) * balance - cost
        balance += trade_pnl  # update balance after trade

        signals.append({
            "ticker": ticker,
            "filing_date": filing_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "signal": signal,
            "return_%": round(ret, 2),
            "reaction": reaction,
            "sentiment_bull": sentiment_bull,
            "sentiment_bear": sentiment_bear,
            "trade_pnl": round(trade_pnl, 2),
            "balance_after_trade": round(balance, 2),
        })

    return signals

def run_backtests(
    bull_threshold, bear_threshold, wait_weeks, hold_weeks,
    initial_balance=100000, transaction_cost=0.001
):
    """
    Run backtests across all combinations of return modes, wait weeks, and hold weeks,
    generating CSV files summarizing trade signals and returns.

    Parameters:
    -----------
    bull_threshold : float
        Threshold ratio for bullish sentiment classification.
    bear_threshold : float
        Threshold ratio for bearish sentiment classification.
    wait_weeks : iterable of int
        Iterable of waiting periods in weeks before entering trades.
    hold_weeks : iterable of int
        Iterable of holding periods in weeks for trades.
    initial_balance : float, optional
        Initial capital for all backtests (default 100,000).
    transaction_cost : float, optional
        Transaction cost per trade side (default 0.001).

    Returns:
    --------
    None
        Saves backtest results as CSV files under 'backtest/' directory.
    """
    # load mdna sentiment and return data
    sentiment_df = pd.read_csv("data/mdna_sentiment_scores.csv", parse_dates=["filing_date"])
    return_modes = {
        1: "return_mode1_signed",
        2: "return_mode2_volume_spike",
        3: "return_mode3_top3_vol_avg"
    }

    # loop through all combinations (meant to backtest and find optimal parameters)
    for i in range(1, 4):  # 3 return modes
        for j in wait_weeks:  # wait length in weeks
            for k in hold_weeks:  # holding length in weeks
                signals = []

                # for each ticker and filing, generate trade signals and calculate returns
                for ticker in sentiment_df["ticker"].unique():
                    price_df = load_price_data(ticker)
                    if price_df is None:
                        continue

                    # iterate over each filing row for this ticker to generate signals
                    ticker_df = sentiment_df[sentiment_df["ticker"] == ticker].copy()
                    ticker_signals = generate_signals_for_ticker(
                        bull_threshold=bull_threshold,
                        bear_threshold=bear_threshold,
                        ticker=ticker,
                        ticker_df=ticker_df,
                        price_df=price_df,
                        return_col=return_modes[i],
                        wait_weeks=j,
                        hold_weeks=k,
                        initial_balance=initial_balance,
                        transaction_cost=transaction_cost
                    )

                    # tag with backtest config
                    for s in ticker_signals:
                        s.update({
                            "mode": i,
                            "wait_weeks": j,
                            "hold_weeks": k
                        })

                    signals.extend(ticker_signals)

                signals_df = pd.DataFrame(signals)
                signals_df.to_csv(f"backtest/mode{i}_wait{j}_hold{k}.csv", index=False)

def analyze_backtests(tickers, start_date, initial_balance):
    """
    Load, analyze, and visualize all backtest CSV files; print top and bottom performance metrics.

    Parameters:
    -----------
    tickers : list of str
        List of tickers analyzed (used for plotting and reporting).
    start_date : str
        Start date for analysis (not directly used here, but kept for consistency).
    initial_balance : float
        Initial capital used in backtests.

    Returns:
    --------
    None
        Displays plots and prints summary statistics.
    """
    master_df = load_backtest_data()
    summary = summarize_backtest_results(master_df)

    # plot performance heatmaps across wait/hold for each mode and metric
    metrics = ["mean_norm_return", "sharpe_ratio", "win_rate"]
    plot_heatmaps(summary, metrics)

    # plot pnl, drawdown, etc. for best config per mode
    for mode in sorted(summary["mode"].unique()):
        best_config = (
            summary[summary["mode"] == mode]
            .sort_values("sharpe_ratio", ascending=False)
            .iloc[0]["config"]
        )
        config_trades = master_df[master_df["config"] == best_config].copy()
        plot_config_timeseries(config_trades, best_config, initial_balance)

    summary = pd.read_csv("backtest/summary_results.csv")

    print_top_bottom_metrics(summary, metric="sharpe_ratio", top_n=10)
    print_top_bottom_metrics(summary, metric="mean_norm_return", top_n=10)

def load_backtest_data():
    """
    Load and concatenate all backtest CSV files into a single DataFrame with added metadata.

    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame of all backtest trades with additional columns like normalized returns and signal types.
    """
    all_paths = glob.glob("backtest/mode*_wait*_hold*.csv")
    all_results = []

    for path in all_paths:
        df = pd.read_csv(path)
        df["config"] = os.path.basename(path).replace(".csv", "")  # e.g., 'mode1_wait0_hold1'
        df["normalized_return"] = df["return_%"] / df["hold_weeks"]
        all_results.append(df)

    master_df = pd.concat(all_results, ignore_index=True)
    master_df["is_long"] = master_df["signal"] == "long"
    master_df["is_short"] = master_df["signal"] == "short"
    return master_df

def summarize_backtest_results(master_df):
    """
    Summarize backtest results by configuration, calculating performance metrics and saving a summary CSV.

    Parameters:
    -----------
    master_df : pandas.DataFrame
        Combined DataFrame of all backtest trades.

    Returns:
    --------
    pandas.DataFrame
        Summary DataFrame with metrics such as trade count, median return, win rate, Sharpe ratio, etc.
    """
    summary = (
        master_df
        .groupby("config", group_keys=False)
        .apply(lambda group: pd.Series({
            "trades": group["return_%"].count(),
            "median_return": group["return_%"].median(),
            "std_return": group["return_%"].std(),
            "win_rate": (group["return_%"] > 0).mean(),
            "mean_norm_return": group["normalized_return"].mean(),
            "sharpe_ratio": group["normalized_return"].mean() / group["normalized_return"].std()
            if group["normalized_return"].std() > 0 else 0,
            "avg_long_return": group.loc[group["is_long"], "return_%"].mean(),
            "avg_short_return": group.loc[group["is_short"], "return_%"].mean()
        }))
        .reset_index()
    )

    summary[["mode", "wait", "hold"]] = summary["config"].str.extract(r"mode(\d+)_wait(\d+)_hold(\d+)")
    summary[["mode", "wait", "hold"]] = summary[["mode", "wait", "hold"]].astype(int)
    summary.to_csv("backtest/summary_results.csv", index=False)
    return summary

def plot_heatmaps(summary, metrics):
    """
    Plot heatmaps of backtest performance metrics across wait and hold week combinations for each mode.

    Parameters:
    -----------
    summary : pandas.DataFrame
        Summary DataFrame containing backtest metrics grouped by config.
    metrics : list of str
        List of metric column names to plot heatmaps for (e.g., ["sharpe_ratio", "win_rate"]).

    Returns:
    --------
    None
        Displays heatmap plots.
    """
    for metric in metrics:
        for mode in sorted(summary["mode"].unique()):
            subset = summary[summary["mode"] == mode]
            pivot_table = subset.pivot(index="wait", columns="hold", values=metric)

            plt.figure(figsize=(10, 7))
            sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="RdYlGn", center=0)
            plt.title(f"{metric.replace('_', ' ').title()} Heatmap — Mode {mode}")
            plt.xlabel("Hold Weeks")
            plt.ylabel("Wait Weeks")
            plt.tight_layout()
            plt.show()

def plot_config_timeseries(config_trades, best_config, initial_balance, benchmark_df=None):
    """
    Plot cumulative PnL, drawdowns, return histograms, and rolling Sharpe ratio for a selected backtest configuration.

    Parameters:
    -----------
    config_trades : pandas.DataFrame
        DataFrame of trades for the selected configuration.
    best_config : str
        Configuration identifier string.
    initial_balance : float
        Starting capital for cumulative return calculations.
    benchmark_df : pandas.DataFrame, optional
        DataFrame with benchmark price data (e.g., SPY), indexed by date, to compare against.

    Returns:
    --------
    None
        Displays multiple performance plots.
    """
    config_trades.sort_values("exit_date", inplace=True)
    cum_pnl = config_trades["trade_pnl"].cumsum()
    exit_dates = pd.to_datetime(config_trades["exit_date"])

    def format_time_axis(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

    # cumulative return percentage for strategy
    cum_return_pct = cum_pnl / initial_balance * 100
    plt.figure(figsize=(12, 6))
    plt.plot(exit_dates, cum_return_pct, label="Strategy", marker="o")

    # plot benchmark comparisons
    if benchmark_df is not None:
        # align benchmark dates to strategy's date range
        benchmark_subset = benchmark_df[(benchmark_df.index >= exit_dates.min()) & (benchmark_df.index <= pd.Timestamp.today())]

        # normalize both SPY and EqualWeighted to 100%
        spy_normalized = (benchmark_subset['SPY'] / benchmark_subset['SPY'].iloc[0]) * 100
        eqw_normalized = (benchmark_subset['EqualWeighted'] / benchmark_subset['EqualWeighted'].iloc[0]) * 100

        plt.plot(benchmark_subset.index, spy_normalized, label="SPY", linestyle="--")
        plt.plot(benchmark_subset.index, eqw_normalized, label="Equal-Weighted", linestyle="--")

    plt.title(f"Cumulative Return (%) Over Time — {best_config}")
    plt.xlabel("Exit Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True)
    format_time_axis(plt.gca())
    plt.tight_layout()
    plt.show()

    # plot drawdown
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    plt.figure(figsize=(12, 6))
    plt.fill_between(exit_dates, drawdown, 0, color='red', alpha=0.3)
    plt.title(f"Drawdown Over Time — {best_config}")
    plt.xlabel("Exit Date")
    plt.ylabel("Drawdown ($)")
    plt.grid(True)
    format_time_axis(plt.gca())
    plt.tight_layout()
    plt.show()

    # histogram of returns
    plt.figure(figsize=(10, 6))
    sns.histplot(config_trades["return_%"], bins=30, kde=True)
    plt.title(f"Histogram of Trade Returns (%) — {best_config}")
    plt.xlabel("Return (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # rolling sharpe ratio
    returns = config_trades["return_%"] / 100
    rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std()
    plt.figure(figsize=(12, 6))
    plt.plot(exit_dates, rolling_sharpe)
    plt.title(f"Rolling Sharpe Ratio (window=20 trades) — {best_config}")
    plt.xlabel("Exit Date")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    format_time_axis(plt.gca())
    plt.tight_layout()
    plt.show()

def print_top_bottom_metrics(summary_df, metric, top_n=10):
    """
    Print the top and bottom configurations ranked by a specified performance metric.

    Parameters:
    -----------
    summary_df : pandas.DataFrame
        DataFrame containing summarized backtest configurations.
    metric : str
        Column name of the metric to rank by (e.g., "sharpe_ratio").
    top_n : int, optional
        Number of top and bottom entries to display (default 10).

    Returns:
    --------
    None
        Prints ranked configurations to the console.
    """
    print(f"\n=== Top {top_n} Configs by {metric} ===")
    top_configs = summary_df.sort_values(metric, ascending=False).head(top_n)
    for _, row in top_configs.iterrows():
        print(
            f"{row['config']:15} | {metric}: {row[metric]:.4f} | Trades: {row['trades']} | Win Rate: {row['win_rate'] * 100:.2f}%")

    print(f"\n=== Bottom {top_n} Configs by {metric} ===")
    bottom_configs = summary_df.sort_values(metric, ascending=True).head(top_n)
    for _, row in bottom_configs.iterrows():
        print(
            f"{row['config']:15} | {metric}: {row[metric]:.4f} | Trades: {row['trades']} | Win Rate: {row['win_rate'] * 100:.2f}%")