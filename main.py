import sys
import nltk

sys.path.insert(0, "src")

from fetch_ticker_data import download_price_data, fetch_and_save_sector_info
from analyze_ticker import full_analysis
from mdna_extractor import download_and_extract_mdna
from mdna_sentiment_and_returns import extract_sentiment_and_reaction_returns
from backtest_runner import run_backtests, analyze_backtests

nltk.download('vader_lexicon', quiet=True)

def main():
    """
    Main script for running the full pipeline of financial data processing, analysis,
    sentiment extraction, and backtesting.

    Workflow:
    1. Downloads historical price data and sector information for a predefined list of tickers.
    2. Performs comprehensive analysis for each ticker (price plots, returns, etc.).
    3. Downloads and extracts MD&A (Management Discussion & Analysis) text from SEC filings.
    4. Extracts sentiment and reaction returns based on the MD&A text.
    5. Runs backtests on trading strategies using sentiment signals and analyzes the results.

    Modules imported:
    - fetch_ticker_data
    - analyze_ticker
    - mdna_extractor
    - mdna_sentiment_and_returns
    - backtest_runner

    Usage:
    Run this script directly. Ensure dependencies are installed and
    './src' is included in the Python path for module imports.

    Author: Aditya Dhanraj
    """

    tickers = [
        "NVDA", "IBM", "TXN", "BAC", "MA", "JNJ", "LLY", "CAT", "DE", "FDX", "T"
    ]
    start_date = "2013-01-01"

    download_price_data(tickers, start_date)
    fetch_and_save_sector_info(tickers) # optional

    '''
    # perform an analysis of each ticker (plot prices, returns, etc.)
    for ticker in tickers:
        full_analysis(ticker)
    '''
    
    # very computation heavy procedure!!!
    download_and_extract_mdna(tickers, filing_type="10-Q", after="2013-01-01")
    extract_sentiment_and_reaction_returns()

    # define base variables for the backtests and run
    bull_threshold = 3
    bear_threshold = 1
    wait_weeks = range(0, 11)
    hold_weeks = range(1, 11)
    initial_balance = 100000
    transaction_cost = 0.001
    run_backtests(bull_threshold, bear_threshold, wait_weeks, hold_weeks, initial_balance, transaction_cost)
    analyze_backtests(tickers, start_date, initial_balance)

if __name__ == "__main__":
    main()