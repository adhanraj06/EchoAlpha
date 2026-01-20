import os

import pandas as pd
import numpy as np


def load_price_data(ticker):
    """Load price data for a given ticker from a CSV file.

    Args:
        ticker (str): Ticker symbol of the stock.

    Returns:
        pandas.DataFrame: Price data for the ticker, with 'Date' as the index.
    """

    try:
        price_df = pd.read_csv(
            f"data/price_data/{ticker}_daily.csv", parse_dates=["Date"]
        )
        price_df.set_index("Date", inplace=True)
        price_df.sort_index(inplace=True)
        return price_df
    except FileNotFoundError:
        print(f"Missing Price Data for {ticker}")
        return None


def generate_signals_for_ticker(
    bull_threshold,
    bear_threshold,
    ticker,
    ticker_df,
    price_df,
    return_col,
    wait_weeks,
    hold_weeks,
    initial_balance=100000,
    transaction_cost=0.001,
    analysis_mode="real",
    strategy="longshort",
    null_seed: int | None = None,
    max_position_size: float = 0.10,
):
    """Generate trading signals for a given ticker based on sentiment and price data.

    Args:
        bull_threshold: Bull threshold for sentiment classification
        bear_threshold: Bear threshold for sentiment classification
        ticker: Corresponding ticker to generate signals for
        ticker_df: Sentiment data for the ticker
        price_df: Price data for the ticker
        return_col: Column name for the return data
        wait_weeks: Wait weeks for signal
        hold_weeks: Hold weeks for signal
        initial_balance: Initial balance for the backtest
        transaction_cost: Transaction cost as a percentage
        analysis_mode: "real" or "null"
        strategy: "long" or "short" or "longshort"
        null_seed: Random seed for null mode (optional)
        max_position_size: Maximum position size as a percentage of balance

    Returns:
        List of trading signals for the given ticker.
    """

    # Classifies a filing as bullish based on sentiment
    def is_bullish(row):
        if row["neg"] == 0:
            return row["pos"] > 0
        return (row["pos"] / row["neg"]) > bull_threshold

    # Classifies a filing as bearish based on sentiment
    def is_bearish(row):
        if row["neg"] == 0:
            return False
        return (row["pos"] / row["neg"]) < bear_threshold

    # Generate signals
    signals = []
    balance = initial_balance
    if null_seed is None:
        base = 0
    else:
        base = int(null_seed)

    seed = (
        hash(
            (
                base,
                ticker,
                bull_threshold,
                bear_threshold,
                return_col,
                wait_weeks,
                hold_weeks,
                strategy,
            )
        )
        & 0xFFFFFFFF
    )
    rng = np.random.default_rng(seed)
    for _, row in ticker_df.iterrows():
        filing_date = pd.to_datetime(row["filing_date"], errors="coerce")
        if pd.isna(filing_date):
            continue

        reaction = row[return_col]
        if pd.isna(reaction):
            continue

        # If reaction = negative and filing = bullish, generate a long signal
        # If reaction = positive and filing = bearish, generate a short signal
        real_direction = None
        sentiment_bull = is_bullish(row)
        sentiment_bear = is_bearish(row)
        if reaction < 0 and sentiment_bull:
            real_direction = "long"
            if strategy == "short":
                continue
        elif reaction > 0 and sentiment_bear:
            real_direction = "short"
            if strategy == "long":
                continue
        else:
            continue

        signal = None
        if analysis_mode == "real":
            signal = real_direction
        elif analysis_mode == "null":
            signal = rng.choice(["long", "short"])

        # Intended dates (calendar-based)
        entry_date = filing_date + pd.Timedelta(weeks=wait_weeks)
        exit_date = entry_date + pd.Timedelta(weeks=hold_weeks)

        # Executed dates (trading-calendar based)
        def next_trading_day(idx, dt):
            pos = idx.searchsorted(dt)
            return None if pos >= len(idx) else idx[pos]

        entry_dt = next_trading_day(price_df.index, entry_date)
        exit_dt = next_trading_day(price_df.index, exit_date)
        if entry_dt is None or exit_dt is None:
            continue
        entry_price = price_df.loc[entry_dt, "Close"]
        exit_price = price_df.loc[exit_dt, "Close"]

        ret = (exit_price - entry_price) / entry_price * 100
        if signal == "short":
            ret *= -1

        # Calculate position size and transaction costs
        signal_numeric = (row["pos"] - row["neg"]) / max(row["pos"] + row["neg"], 1)
        position_size = balance * min(abs(signal_numeric), max_position_size)
        cost = position_size * transaction_cost * 2
        trade_pnl = (ret / 100) * position_size - cost
        balance += trade_pnl

        signals.append(
            {
                "ticker": ticker,
                "filing_date": filing_date,
                "entry_date": entry_dt,
                "exit_date": exit_dt,
                "signal": signal,
                "signal_numeric": signal_numeric,
                "return_%": round(ret, 2),
                "reaction": reaction,
                "sentiment_bull": sentiment_bull,
                "sentiment_bear": sentiment_bear,
                "trade_pnl": round(trade_pnl, 2),
                "balance_after_trade": round(balance, 2),
            }
        )

    return signals


def run_backtests(
    bull_thresholds,
    bear_thresholds,
    wait_weeks,
    hold_weeks,
    lambdas,
    robustness_ratio=1,
    initial_balance=100000,
    transaction_cost=0.001,
    analysis_mode="real",
    strategy="longshort",
    save_path="backtest/grid_results.csv",
    null_seed: int | None = None,
):
    """Run backtests for different configurations.

    Args:
        bull_thresholds: List of bull threshold values to test
        bear_thresholds: List of bear threshold values to test
        wait_weeks: List of wait week values to test
        hold_weeks: List of hold week values to test
        lambdas: List of lambda values to test
        robustness_ratio: Ratio to multiply st dev by for robust EAE calculation
        initial_balance: Initial balance for the backtest
        transaction_cost: Transaction cost as a percentage
        analysis_mode: "real" or "null"
        strategy: "l" or "s" or "longshort" to specify the strategy
        save_path: Path to save the results
        null_seed: Random seed for null mode (optional)

    Returns:
        DataFrame containing the results of the backtest
    """

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sentiment_df = pd.read_csv(
        "data/mdna_sentiment_scores.csv", parse_dates=["filing_date"]
    )

    return_modes = {
        1: "return_mode1_signed",
        2: "return_mode2_volume_spike",
        3: "return_mode3_top3_vol_avg",
    }

    # Load price and sentiment data once (big speedup, no result change)
    tickers = sorted(sentiment_df["ticker"].unique())
    price_cache = {t: load_price_data(t) for t in tickers}
    sent_by_ticker = {t: df.copy() for t, df in sentiment_df.groupby("ticker")}
    max_position_size = (100 // len(tickers)) / 100

    results = []

    for bull in bull_thresholds:
        for bear in bear_thresholds:
            for mode_id, return_col in return_modes.items():
                for w in wait_weeks:
                    for h in hold_weeks:
                        # collect trades across tickers for THIS config
                        signals = []
                        for ticker in tickers:
                            price_df = price_cache.get(ticker)
                            if price_df is None:
                                continue
                            ticker_df = sent_by_ticker.get(ticker)
                            if ticker_df is None:
                                continue

                            ticker_signals = generate_signals_for_ticker(
                                bull,
                                bear,
                                ticker,
                                ticker_df,
                                price_df,
                                return_col,
                                w,
                                h,
                                initial_balance,
                                transaction_cost,
                                analysis_mode,
                                strategy,
                                null_seed=null_seed,
                                max_position_size=max_position_size,
                            )
                            signals.extend(ticker_signals)

                        trades_df = pd.DataFrame(signals)

                        metrics = summarize_trades_df(trades_df, initial_balance)

                        row = {
                            "bull_threshold": bull,
                            "bear_threshold": bear,
                            "return_mode": mode_id,
                            "wait_weeks": w,
                            "hold_weeks": h,
                            "analysis_mode": analysis_mode,
                            "strategy": strategy,
                            "null_seed": null_seed,
                            **metrics,
                        }
                        results.append(row)

    results_df = pd.DataFrame(results)

    # Estimate n0 from the grid itself and compute EAE for each lambda (long format)
    n0_hat = int(estimate_n0_from_bins(results_df, metric="cum_return_pct"))
    min_trades = max(20, n0_hat // 2)
    results_df["n0_hat"] = n0_hat

    eae_means: list[float] = []
    eae_stds: list[float] = []

    for row in results_df.itertuples(index=False):
        trades = getattr(row, "trades", None)
        trades = int(trades) if pd.notna(trades) else None

        eae_vals = []
        for lam in lambdas:
            eae = compute_eae(
                cum_return_pct=getattr(row, "cum_return_pct"),
                max_drawdown_pct=getattr(row, "max_drawdown"),
                trades=trades,
                lam=float(lam),
                n0=n0_hat,
                min_trades=min_trades,
            )
            eae_vals.append(float(eae))

        eae_means.append(float(np.mean(eae_vals)))
        eae_stds.append(float(np.std(eae_vals, ddof=1)))

    results_df["eae_mean"] = eae_means
    results_df["eae_std"] = eae_stds

    results_df["eae"] = (
        results_df["eae_mean"] - float(robustness_ratio) * results_df["eae_std"]
    )

    if save_path is not None:
        results_df.to_csv(save_path, index=False)

    return results_df


def summarize_trades_df(trades_df: pd.DataFrame, initial_balance: float) -> dict:
    """Summarize trade results for a single configuration.

    Args:
        trades_df: DataFrame with trade results
        initial_balance: Initial balance for the backtest

    Returns:
        Dictionary with summary metrics
    """

    if trades_df is None or trades_df.empty:
        return {
            "trades": 0,
            "bull_trades": 0,
            "bear_trades": 0,
            "long_signal_count": 0,
            "short_signal_count": 0,
            "avg_long_return": np.nan,
            "avg_short_return": np.nan,
            "long_signal_numeric_mean": np.nan,
            "long_signal_numeric_std": 0.0,
            "short_signal_numeric_mean": np.nan,
            "short_signal_numeric_std": 0.0,
            "total_signal_numeric_mean": np.nan,
            "total_signal_numeric_std": 0.0,
            "long_signal_numeric_count": 0,
            "short_signal_numeric_count": 0,
            "long_signal_ic": np.nan,
            "short_signal_ic": np.nan,
            "trades_ic": np.nan,
            "bull_win_rate": np.nan,
            "bear_win_rate": np.nan,
            "bull_return": np.nan,
            "bear_return": np.nan,
            "long_signal_win_rate": np.nan,
            "short_signal_win_rate": np.nan,
            "cum_return_pct": 0.0,
            "long_return_pct": 0.0,
            "short_return_pct": 0.0,
            "median_return": np.nan,
            "std_return": 0.0,
            "win_rate": np.nan,
            "max_drawdown": np.nan,
        }

    g = trades_df.copy()

    g["is_long"] = g["signal"] == "long"
    g["is_short"] = g["signal"] == "short"

    # Safe helpers
    def safe_mean(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean()) if len(x) else np.nan

    def safe_std(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.std(ddof=1)) if len(x) >= 2 else 0.0

    def safe_corr(x: pd.Series, y: pd.Series, min_n: int = 2) -> float:
        xy = pd.concat([x, y], axis=1).dropna()
        if len(xy) < min_n:
            return np.nan
        a, b = xy.iloc[:, 0], xy.iloc[:, 1]
        if a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
            return np.nan
        return float(a.corr(b))

    # Core metrics
    trades = int(g["return_%"].count())
    bull_trades = int(g["sentiment_bull"].sum()) if "sentiment_bull" in g.columns else 0
    bear_trades = int(g["sentiment_bear"].sum()) if "sentiment_bear" in g.columns else 0
    long_count = int(g["is_long"].sum())
    short_count = int(g["is_short"].sum())
    avg_long_return = safe_mean(g.loc[g["is_long"], "return_%"])
    avg_short_return = safe_mean(g.loc[g["is_short"], "return_%"])
    long_sig = g.loc[g["is_long"], "signal_numeric"]
    short_sig = g.loc[g["is_short"], "signal_numeric"]
    all_sig = g["signal_numeric"]
    long_ic = safe_corr(g.loc[g["is_long"], "return_%"], long_sig)
    short_ic = safe_corr(g.loc[g["is_short"], "return_%"], short_sig)
    trades_ic = safe_corr(g["return_%"], all_sig)
    bull_mask = (
        g.get("sentiment_bull", False).astype(bool)
        if "sentiment_bull" in g.columns
        else pd.Series(False, index=g.index)
    )
    bear_mask = (
        g.get("sentiment_bear", False).astype(bool)
        if "sentiment_bear" in g.columns
        else pd.Series(False, index=g.index)
    )
    bull_win_rate = safe_mean(g.loc[bull_mask, "return_%"] > 0)
    bear_win_rate = safe_mean(g.loc[bear_mask, "return_%"] > 0)
    bull_return = safe_mean(g.loc[bull_mask, "return_%"])
    bear_return = safe_mean(g.loc[bear_mask, "return_%"])
    long_win_rate = safe_mean(g.loc[g["is_long"], "return_%"] > 0)
    short_win_rate = safe_mean(g.loc[g["is_short"], "return_%"] > 0)
    cum_return_pct = float(g["trade_pnl"].sum() / initial_balance * 100)
    long_return_pct = float(
        g.loc[g["is_long"], "trade_pnl"].sum() / initial_balance * 100
    )
    short_return_pct = float(
        g.loc[g["is_short"], "trade_pnl"].sum() / initial_balance * 100
    )
    median_return = float(g["return_%"].median())
    std_return = safe_std(g["return_%"])
    win_rate = safe_mean(g["return_%"] > 0)
    wins = g.loc[g["trade_pnl"] > 0, "trade_pnl"].sum()
    losses = g.loc[g["trade_pnl"] < 0, "trade_pnl"].sum()
    profit_factor = np.nan if losses == 0 else float(wins / abs(losses))
    g = g.sort_values("exit_date")
    equity = initial_balance + g["trade_pnl"].cumsum()
    peak = equity.cummax()
    dd = equity.div(peak).replace([np.inf, -np.inf], np.nan) - 1.0
    max_dd = float(dd.min() * 100)

    return {
        "trades": trades,
        "bull_trades": bull_trades,
        "bear_trades": bear_trades,
        "long_signal_count": long_count,
        "short_signal_count": short_count,
        "avg_long_return": avg_long_return,
        "avg_short_return": avg_short_return,
        "long_signal_numeric_mean": safe_mean(long_sig),
        "long_signal_numeric_std": safe_std(long_sig),
        "short_signal_numeric_mean": safe_mean(short_sig),
        "short_signal_numeric_std": safe_std(short_sig),
        "total_signal_numeric_mean": safe_mean(all_sig),
        "total_signal_numeric_std": safe_std(all_sig),
        "long_signal_numeric_count": int(long_sig.count()),
        "short_signal_numeric_count": int(short_sig.count()),
        "long_signal_ic": long_ic,
        "short_signal_ic": short_ic,
        "trades_ic": trades_ic,
        "bull_win_rate": bull_win_rate,
        "bear_win_rate": bear_win_rate,
        "bull_return": bull_return,
        "bear_return": bear_return,
        "long_signal_win_rate": long_win_rate,
        "short_signal_win_rate": short_win_rate,
        "cum_return_pct": cum_return_pct,
        "long_return_pct": long_return_pct,
        "short_return_pct": short_return_pct,
        "median_return": median_return,
        "std_return": std_return,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
    }


def estimate_n0_from_bins(
    df: pd.DataFrame,
    *,
    metric: str = "cum_return_pct",
    bins=(0, 5, 10, 15, 20, 30, 50, 100, 200, 500),
    tol: float = 0.10,
    tail_bins: int = 2,
) -> int:
    """
    Estimate n0 (reliability saturation) from historical backtest bins.

    Args:
        df: DataFrame with backtest results (must have "trades" and metric columns)
        metric: column name to compute std over (e.g., "cum_return_pct")
        bins: trade count bins for grouping
        tol: tolerance for "close enough to plateau" (as fraction)
        tail_bins: number of top bins to average for plateau definition

    Returns:
        Estimated n0 (reliability saturation trades)
    """

    x = df.copy()
    x = x[np.isfinite(x["trades"]) & np.isfinite(x[metric])]
    if x.empty:
        return 20

    x["trade_bin"] = pd.cut(x["trades"], bins=bins, include_lowest=True)
    s = x.groupby("trade_bin", observed=True)[metric].std().dropna()
    if len(s) == 0:
        return 20

    plateau = float(s.tail(tail_bins).mean())
    if not np.isfinite(plateau) or plateau <= 0:
        return 20

    thresh = plateau * (1.0 + tol)

    # pick earliest bin whose dispersion is ~plateau
    for interval, val in s.items():
        if np.isfinite(val) and val <= thresh:
            return int(interval.right)

    # else fallback to largest trade bin edge
    return int(s.index[-1].right)


def compute_eae(
    cum_return_pct: float,
    max_drawdown_pct: float,
    trades: int,
    *,
    lam: float = 0.5,
    n0: int = 20,
    min_trades: int = 10,
) -> float:
    """Compute Evidence-Adjusted Edge (EAE) score for a backtest configuration.

    Args:
        cum_return_pct: Cumulative return percentage
        max_drawdown_pct: Maximum drawdown percentage
        trades: Number of trades
        lam: Drawdown penalty weight
        n0: Reliability saturation trades
        min_trades: Hard floor to avoid tiny-sample configs

    Returns:
        Score = (Return - lam * |MaxDD|) * (1 - exp(-n/n0))
    """

    # basic validity
    if trades is None or trades < min_trades:
        return float("-inf")

    if cum_return_pct is None or not np.isfinite(cum_return_pct):
        return float("-inf")

    if max_drawdown_pct is None or not np.isfinite(max_drawdown_pct):
        return float("-inf")

    # guard n0
    n0 = int(n0)
    if n0 <= 0:
        n0 = 1

    dd_abs = abs(float(max_drawdown_pct))
    reliability = 1.0 - float(np.exp(-float(trades) / float(n0)))

    return (float(cum_return_pct) - float(lam) * dd_abs) * reliability
