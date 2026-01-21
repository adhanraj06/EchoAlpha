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
    analysis_mode="real",
    strategy="longshort",
    null_seed: int | None = None,
):
    """Generate trade intents for a ticker (no sizing, no PnL)."""

    def is_bullish(row):
        s = row.get("finbert_score", np.nan)
        if pd.isna(s):
            return False
        return float(s) >= float(bull_threshold)

    def is_bearish(row):
        s = row.get("finbert_score", np.nan)
        if pd.isna(s):
            return False
        return float(s) <= -float(bear_threshold)

    trades = []

    base = 0 if null_seed is None else int(null_seed)
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

    def next_trading_day(idx, dt):
        pos = idx.searchsorted(dt)
        return None if pos >= len(idx) else idx[pos]

    for _, row in ticker_df.iterrows():
        filing_date = pd.to_datetime(row.get("filing_date", None), errors="coerce")
        if pd.isna(filing_date):
            continue

        reaction = row.get(return_col, np.nan)
        if pd.isna(reaction):
            continue

        sentiment_bull = is_bullish(row)
        sentiment_bear = is_bearish(row)

        real_direction = None
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

        if analysis_mode == "real":
            signal = real_direction
        elif analysis_mode == "null":
            signal = rng.choice(["long", "short"])
        else:
            continue

        entry_date = filing_date + pd.Timedelta(weeks=wait_weeks)
        exit_date = entry_date + pd.Timedelta(weeks=hold_weeks)

        entry_dt = next_trading_day(price_df.index, entry_date)
        exit_dt = next_trading_day(price_df.index, exit_date)
        if entry_dt is None or exit_dt is None:
            continue

        entry_price = float(price_df.loc[entry_dt, "Close"])
        exit_price = float(price_df.loc[exit_dt, "Close"])
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
            continue

        raw_ret = (exit_price - entry_price) / entry_price * 100.0

        trades.append(
            {
                "ticker": ticker,
                "filing_date": filing_date,
                "entry_date": entry_dt,
                "exit_date": exit_dt,
                "signal": signal,
                "signal_numeric": float(row.get("finbert_score", 0.0)),
                "raw_return_%": float(raw_ret),
                "reaction": float(reaction),
                "sentiment_bull": bool(sentiment_bull),
                "sentiment_bear": bool(sentiment_bear),
                "entry_price": entry_price,
                "exit_price": exit_price,
            }
        )

    return trades


def simulate_portfolio_events(
    trades_df: pd.DataFrame,
    *,
    initial_balance: float = 100000.0,
    transaction_cost: float = 0.001,
    max_gross: float = 1.0,
    max_net: float = 0.3,
    per_position_cap: float = 0.05,
) -> pd.DataFrame:
    """
    Portfolio simulation with entry/exit events (no daily MTM).

    Expects trades_df to have:
        entry_date, exit_date, signal ("long"/"short"), raw_return_%
    """

    if trades_df is None or trades_df.empty:
        return trades_df

    required = {"entry_date", "exit_date", "signal", "raw_return_%"}
    missing = required - set(trades_df.columns)
    if missing:
        raise ValueError(f"trades_df missing required columns: {sorted(missing)}")

    g = trades_df.copy()

    g["entry_date"] = pd.to_datetime(g["entry_date"], errors="coerce")
    g["exit_date"] = pd.to_datetime(g["exit_date"], errors="coerce")
    g["signal"] = g["signal"].astype(str).str.lower()

    # Drop broken rows
    g = g.dropna(subset=["entry_date", "exit_date", "raw_return_%"])
    g = g[g["signal"].isin(["long", "short"])]
    g = g[g["exit_date"] >= g["entry_date"]]

    if g.empty:
        return g

    g = g.sort_values(["entry_date", "exit_date", "ticker"]).reset_index(drop=True)

    # Output columns (created if missing)
    out_cols = [
        "position_weight",
        "position_size",
        "entry_cost",
        "exit_cost",
        "return_%",
        "trade_pnl",
        "balance_after_entry",
        "balance_after_trade",
    ]
    for c in out_cols:
        if c not in g.columns:
            g[c] = np.nan

    # Build events (exit before entry on same day)
    # event_type: 0 = exit, 1 = entry
    events: list[tuple[pd.Timestamp, int, int]] = []
    for i in range(len(g)):
        events.append((g.at[i, "exit_date"], 0, i))
        events.append((g.at[i, "entry_date"], 1, i))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    balance = float(initial_balance)

    # Store open positions by trade idx
    # We keep entry notional FIXED and use it at exit.
    open_pos: dict[int, dict] = {}

    def exposures() -> tuple[float, float]:
        gross = 0.0
        net = 0.0
        for p in open_pos.values():
            w = float(p["w"])
            gross += abs(w)
            net += w
        return gross, net

    for dt, event_type, idx in events:
        row = g.loc[idx]

        if event_type == 0:
            # EXIT
            if idx not in open_pos:
                continue

            p = open_pos[idx]
            w = float(p["w"])
            notional = float(p["notional"])
            entry_cost = float(p["entry_cost"])

            raw_ret = float(row["raw_return_%"]) / 100.0
            signed_ret = raw_ret if row["signal"] == "long" else -raw_ret

            exit_cost = notional * float(transaction_cost)
            pnl_gross = signed_ret * notional
            trade_pnl = pnl_gross - entry_cost - exit_cost

            balance += trade_pnl

            g.at[idx, "position_weight"] = w
            g.at[idx, "position_size"] = notional
            g.at[idx, "entry_cost"] = entry_cost
            g.at[idx, "exit_cost"] = exit_cost
            g.at[idx, "return_%"] = signed_ret * 100.0
            g.at[idx, "trade_pnl"] = trade_pnl
            g.at[idx, "balance_after_trade"] = balance

            del open_pos[idx]
            continue

        # ENTRY
        if idx in open_pos:
            continue

        gross, net = exposures()

        gross_headroom = float(max_gross) - gross
        if gross_headroom <= 0.0:
            continue

        # Desired absolute weight limited by per-position cap + remaining gross
        w_abs = min(float(per_position_cap), gross_headroom)
        if w_abs <= 0.0:
            continue

        # Enforce net constraint using "net after trade"
        if row["signal"] == "long":
            w = w_abs
            if abs(net + w) > float(max_net):
                # clamp to max allowed
                w = max(0.0, float(max_net) - net)
        else:
            w = -w_abs
            if abs(net + w) > float(max_net):
                # clamp to max allowed (negative)
                w = -max(0.0, net + float(max_net))

        if abs(w) <= 0.0:
            continue

        # FIXED notional at entry (based on current equity at time of entry)
        notional = abs(w) * balance
        if not np.isfinite(notional) or notional <= 0.0:
            continue

        entry_cost = notional * float(transaction_cost)

        open_pos[idx] = {
            "w": float(w),
            "notional": float(notional),
            "entry_cost": float(entry_cost),
        }

        g.at[idx, "position_weight"] = float(w)
        g.at[idx, "position_size"] = float(notional)
        g.at[idx, "entry_cost"] = float(entry_cost)
        g.at[idx, "balance_after_entry"] = balance

    return g


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
    max_gross: float = 1.0,
    max_net: float = 0.3,
    per_position_cap: float = 0.05,
):
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

    tickers = sorted(sentiment_df["ticker"].unique())
    price_cache = {t: load_price_data(t) for t in tickers}
    sent_by_ticker = {t: df.copy() for t, df in sentiment_df.groupby("ticker")}

    results = []

    for bull in bull_thresholds:
        for bear in bear_thresholds:
            for mode_id, return_col in return_modes.items():
                for w in wait_weeks:
                    for h in hold_weeks:
                        trades = []
                        for ticker in tickers:
                            price_df = price_cache.get(ticker)
                            if price_df is None:
                                continue
                            ticker_df = sent_by_ticker.get(ticker)
                            if ticker_df is None:
                                continue

                            ticker_trades = generate_signals_for_ticker(
                                bull,
                                bear,
                                ticker,
                                ticker_df,
                                price_df,
                                return_col,
                                w,
                                h,
                                analysis_mode=analysis_mode,
                                strategy=strategy,
                                null_seed=null_seed,
                            )
                            trades.extend(ticker_trades)

                        trades_df = pd.DataFrame(trades)

                        # portfolio simulation (overlap-aware sizing)
                        sim_df = simulate_portfolio_events(
                            trades_df,
                            initial_balance=float(initial_balance),
                            transaction_cost=float(transaction_cost),
                            max_gross=float(max_gross),
                            max_net=float(max_net),
                            per_position_cap=float(per_position_cap),
                        )

                        metrics = summarize_trades_df(sim_df, float(initial_balance))

                        row = {
                            "bull_threshold": bull,
                            "bear_threshold": bear,
                            "return_mode": mode_id,
                            "wait_weeks": w,
                            "hold_weeks": h,
                            "analysis_mode": analysis_mode,
                            "strategy": strategy,
                            "null_seed": null_seed,
                            "max_gross": max_gross,
                            "max_net": max_net,
                            "per_position_cap": per_position_cap,
                            **metrics,
                        }
                        results.append(row)

    results_df = pd.DataFrame(results)

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
    results_df["eae"] = results_df["eae_mean"] - float(robustness_ratio) * results_df["eae_std"]

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
