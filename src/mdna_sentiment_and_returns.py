import os
import pandas as pd
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_earnings_reaction_returns(ticker, filing_date_str, price_dir="data/price_data", lookback=14):
    """
    Estimate multiple types of earnings reaction returns for a given ticker within a lookback window before the filing date.

    Parameters:
    -----------
    ticker : str
        The stock ticker symbol (e.g., "AAPL").
    filing_date_str : str
        Filing date in 'YYYY-MM-DD' format corresponding to the earnings report date.
    price_dir : str, optional
        Directory path where daily price CSV files are stored. Each file is expected to be named '{ticker}_daily.csv'.
        Default is "data/price_data".
    lookback : int, optional
        Number of trading days to look back from the filing date (inclusive) to calculate reaction returns.
        Default is 14.

    Returns:
    --------
    tuple of three floats or Nones:
        - mode1: Signed largest single-day return in the lookback window (percent).
        - mode2: Return on the day with the largest absolute volume percentage change.
        - mode3: Average return across the top 3 highest volume days in the window.

        If price data is missing or insufficient, returns a tuple of (None, None, None).
    """
    path = os.path.join(price_dir, f"{ticker}_daily.csv")
    if not os.path.isfile(path):
        print(f"Price Data Not Found for {ticker}")
        return [None] * 4

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    df["Return"] = df["Close"].pct_change() * 100  # percent return

    filing_date = pd.to_datetime(filing_date_str)

    # use previous trading day if filing date is not in index
    if filing_date not in df.index:
        try:
            filing_date = df.index[df.index.get_loc(filing_date, method="pad")]
        except KeyError:
            return [None] * 4

    # get lookback window (includes filing date)
    window_df = df.loc[:filing_date].tail(lookback + 1).dropna(subset=["Return"])
    if window_df.empty:
        return [None] * 4

    # mode 1: signed version of largest move
    mode1 = round(window_df["Return"].iloc[window_df["Return"].abs().argmax()], 2)

    # mode 2: return on day with the largest volume % change
    vol_idx = window_df["Volume"].pct_change().abs().idxmax()
    mode2 = round(window_df.loc[vol_idx, "Return"], 2)

    # mode 3: average return across top 3 volume days
    top_vol = window_df.sort_values("Volume", ascending=False).head(3)
    mode3 = round(top_vol["Return"].mean(), 2)

    return mode1, mode2, mode3

def extract_sentiment_and_reaction_returns(
    mdna_dir="data/mdna",
    price_dir="data/price_data",
    output_csv="data/mdna_sentiment_scores.csv"
):
    """
    Analyze MD&A text sentiment and compute earnings reaction returns for multiple tickers and filing dates.

    This function processes all text files (each representing a 10-Q filing) inside subdirectories of `mdna_dir`,
    calculates sentiment scores using NLTK's VADER sentiment analyzer, and estimates earnings reaction returns
    by calling `get_earnings_reaction_returns`. Results are saved as a CSV file.

    Parameters:
    -----------
    mdna_dir : str, optional
        Directory containing subfolders named after tickers. Each subfolder contains text files for filings.
        Default is "data/mdna".
    price_dir : str, optional
        Directory containing daily price CSV files named '{ticker}_daily.csv'.
        Default is "data/price_data".
    output_csv : str, optional
        File path where the combined sentiment and return scores CSV will be saved.
        Default is "data/mdna_sentiment_scores.csv".

    Returns:
    --------
    None
        Writes a CSV file with the following columns per ticker and filing date:
        'ticker', 'filing_date', 'return_mode1_signed', 'return_mode2_volume_spike',
        'return_mode3_top3_vol_avg', 'compound', 'neg', 'neu', 'pos'.
    """
    sia = SentimentIntensityAnalyzer()
    results = []

    # loop through each ticker folder
    for ticker in os.listdir(mdna_dir):
        ticker_dir = os.path.join(mdna_dir, ticker)
        if not os.path.isdir(ticker_dir):
            continue

        # loop through each text file (each represents a 10-q)
        for filename in tqdm(os.listdir(ticker_dir), desc=f"processing {ticker}"):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(ticker_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                continue  # skip empty files

            scores = sia.polarity_scores(text)
            date_str = filename.replace(".txt", "")

            # get all 3 return variants
            r1, r2, r3 = get_earnings_reaction_returns(ticker, date_str, price_dir=price_dir)

            results.append({
                "ticker": ticker,
                "filing_date": date_str,
                "return_mode1_signed": r1,
                "return_mode2_volume_spike": r2,
                "return_mode3_top3_vol_avg": r3,
                "compound": scores["compound"],
                "neg": scores["neg"],
                "neu": scores["neu"],
                "pos": scores["pos"]
            })

    # save dataframe to csv
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)