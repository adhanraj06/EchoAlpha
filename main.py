import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nltk
import pandas as pd

from src.fetch_ticker_data import download_price_data, fetch_and_save_sector_info
from src.analyze_ticker import full_analysis
from src.mdna_extractor import download_and_extract_mdna
from src.mdna_sentiment_and_returns import extract_sentiment_and_reaction_returns
from src.backtest_runner import run_backtests
from src.config_study import run_config_study_bundle

nltk.download("vader_lexicon", quiet=True)


def main():
    """Main function to run the entire pipeline of data processing, analysis,
    sentiment extraction, backtesting, and config study.

    Args:
        None

    Returns:
        None
    """

    results_dir = "backtest/"
    analysis_dir = "ticker_analysis/"
    study_dir = "study/"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(study_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "run_summary.txt")
    with open(summary_path, "w") as f:
        f.write("EchoAlpha Run Summary\n")
        f.write("=" * 50 + "\n\n")

    tickers = ["NVDA", "IBM", "TXN", "BAC", "MA", "JNJ", "LLY", "CAT", "DE", "FDX", "T"]
    start_date = "2013-01-01"

    print(f"[1/12] Downloading historical price data for {len(tickers)} tickers")
    t0 = time.time()
    download_price_data(tickers, start_date)
    print(f"[Timing] Data acquisition: {time.time() - t0:.1f}s")

    print(f"[2/12] Fetching sector info for tickers")
    t0 = time.time()
    fetch_and_save_sector_info(tickers)
    print(f"[Timing] Sector info: {time.time() - t0:.1f}s")

    for ticker in tickers:
        print(f"[3/12] Performing analysis for {ticker}")
        t0 = time.time()
        full_analysis(ticker, out_dir=analysis_dir)
        print(f"[Timing] Analysis for {ticker}: {time.time() - t0:.1f}s")

    if not os.path.exists("data/mdna/"):
        print(f"[4/12] Downloading and extracting MD&A text from SEC filings")
        t0 = time.time()
        download_and_extract_mdna(tickers, filing_type="10-Q", after=start_date)
        print(f"[Timing] MD&A extraction: {time.time() - t0:.1f}s")

    if not os.path.exists("data/mdna_sentiment_scores.csv"):
        print("[5/12] Extracting sentiment scores and reaction returns via MD&A")
        t0 = time.time()
        extract_sentiment_and_reaction_returns()
        print(f"[Timing] Sentiment extraction: {time.time() - t0:.1f}s")

    bull_thresholds = range(2, 6)
    bear_thresholds = range(0, 4)
    wait_weeks = range(0, 11)
    hold_weeks = range(1, 11)
    lambdas = [0.2, 0.4, 0.6, 0.8, 1.0]
    robustness_ratio = 1.0
    initial_balance = 100000
    transaction_cost = 0.001

    if not os.path.exists(os.path.join(results_dir, "real_longshort_grid_results.csv")):
        print("[6/12] Running L/S backtests with sentiment-based signals")
        t0 = time.time()
        analysis_mode = "real"
        strategy = "longshort"
        save_path = os.path.join(
            results_dir, f"{analysis_mode}_{strategy}_grid_results.csv"
        )
        real_ls_summary = run_backtests(
            bull_thresholds,
            bear_thresholds,
            wait_weeks,
            hold_weeks,
            lambdas,
            robustness_ratio,
            initial_balance,
            transaction_cost,
            analysis_mode,
            strategy,
            save_path,
        )
        print(f"[Timing] L/S backtest: {time.time() - t0:.1f}s")

        print("[7/12] Writing L/S backtest results to summary file")
        with open(summary_path, "a") as f:
            f.write("\nL/S Backtest Results (Top 10):\n")
            real_ls_summary = real_ls_summary.sort_values("eae", ascending=False).head(
                10
            )
            f.write(real_ls_summary.to_string(index=False))
            f.write("\n")
    else:
        real_ls_summary = pd.read_csv(
            os.path.join(results_dir, "real_longshort_grid_results.csv")
        )

    if not os.path.exists(os.path.join(results_dir, "real_long_grid_results.csv")):
        print("[8/12] Running Long-only backtests with sentiment-based signals")
        t0 = time.time()
        analysis_mode = "real"
        strategy = "long"
        save_path = os.path.join(
            results_dir, f"{analysis_mode}_{strategy}_grid_results.csv"
        )
        real_long_summary = run_backtests(
            bull_thresholds,
            bear_thresholds,
            wait_weeks,
            hold_weeks,
            lambdas,
            robustness_ratio,
            initial_balance,
            transaction_cost,
            analysis_mode,
            strategy,
            save_path,
        )
        print(f"[Timing] Long-only backtest: {time.time() - t0:.1f}s")

        print("[9/12] Writing Long-only backtest results to summary file")
        with open(summary_path, "a") as f:
            f.write("\nLong-only Backtest Results (Top 10):\n")
            real_long_summary = real_long_summary.sort_values(
                "eae", ascending=False
            ).head(10)
            f.write(real_long_summary.to_string(index=False))
            f.write("\n")
    else:
        real_long_summary = pd.read_csv(
            os.path.join(results_dir, "real_long_grid_results.csv")
        )

    if not os.path.exists(os.path.join(results_dir, "null_grid_results.csv")):
        print("[10/12] Running null backtests")
        t0 = time.time()
        num_null_runs = 100
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        tasks = [
            (
                i + 1,
                bull_thresholds,
                bear_thresholds,
                wait_weeks,
                hold_weeks,
                lambdas,
                robustness_ratio,
                initial_balance,
                transaction_cost,
            )
            for i in range(num_null_runs)
        ]
        all_null_summaries = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_null_backtest, t) for t in tasks]
            for fut in as_completed(futures):
                all_null_summaries.append(fut.result())
        print(f"[Timing] Null backtest pool: {time.time() - t0:.1f}s")

        print("[11/12] Writing null backtest results to summary file")
        all_null_summary = pd.concat(all_null_summaries, ignore_index=True)
        all_null_summary.to_csv(
            os.path.join(results_dir, "null_grid_results.csv"), index=False
        )
        with open(summary_path, "a") as f:
            f.write("\nNull Backtest Results (Top 10):\n")
            all_null_summary = all_null_summary.sort_values(
                "eae", ascending=False
            ).head(10)
            f.write(all_null_summary.to_string(index=False))
            f.write("\n")
    else:
        all_null_summary = pd.read_csv(
            os.path.join(results_dir, "null_grid_results.csv")
        )

    if not os.path.exists("backtest/study"):
        print("[12/12] Running configuration study bundle on results")
        t0 = time.time()
        run_config_study_bundle(
            real_ls_summary,
            real_long_summary,
            out_dir="study",
            null_df=all_null_summary,
        )
        print(f"[Timing] Visualization and analysis: {time.time() - t0:.1f}s")


def run_null_backtest(args):
    """
    Run a single null backtest with the given parameters.

    Args:
        args: Tuple containing (run_idx, bull_thresholds, bear_thresholds,
        wait_weeks, hold_weeks, lambdas, robustness_ratio, initial_balance, transaction_cost)

    Returns:
        pandas.DataFrame: Backtest results for this run
    """

    (
        run_idx,
        bull_thresholds,
        bear_thresholds,
        wait_weeks,
        hold_weeks,
        lambdas,
        robustness_ratio,
        initial_balance,
        transaction_cost,
    ) = args

    df = run_backtests(
        bull_thresholds,
        bear_thresholds,
        wait_weeks,
        hold_weeks,
        lambdas,
        robustness_ratio,
        initial_balance,
        transaction_cost,
        analysis_mode="null",
        strategy="longshort",
        save_path=None,
        null_seed=run_idx,
    )
    return df


if __name__ == "__main__":
    main()
