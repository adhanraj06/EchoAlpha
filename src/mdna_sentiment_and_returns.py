import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


def get_earnings_reaction_returns(
    ticker, filing_date_str, price_dir="data/price_data", lookback=14
):
    """
    Estimate multiple returns for a ticker within a lookback window before filing.

    Args:
        ticker: Ticker symbol (e.g., "AAPL")
        filing_date_str: Filing date in 'YYYY-MM-DD' format (the earnings report date)
        price_dir: Directory path where daily price CSV files are stored
        lookback: Number of trading days to look back from filing for reaction returns

    Returns:
        tuple of three floats or Nones:
            - mode1: Signed largest single-day return in the lookback window (percent).
            - mode2: Return on day with the largest absolute volume percentage change.
            - mode3: Average return across top 3 highest volume days in the window.

        If price data is missing or insufficient, returns a tuple of (None, None, None).
    """

    path = os.path.join(price_dir, f"{ticker}_daily.csv")
    if not os.path.isfile(path):
        print(f"Price Data Not Found for {ticker}")
        return [None] * 3
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    df["Return"] = df["Close"].pct_change() * 100  # percent return

    # Use previous trading day if filing date is not in index
    filing_date = pd.to_datetime(filing_date_str)
    if filing_date not in df.index:
        try:
            filing_date = df.index[df.index.get_loc(filing_date, method="pad")]
        except KeyError:
            return [None] * 3

    # Get lookback window (includes filing date)
    window_df = df.loc[:filing_date].tail(lookback + 1).dropna(subset=["Return"])
    if window_df.empty:
        return [None] * 3

    # Mode 1: signed version of largest move
    mode1 = round(window_df["Return"].iloc[window_df["Return"].abs().argmax()], 2)

    # Mode 2: return on day with the largest volume % change
    vol_idx = window_df["Volume"].pct_change().abs().idxmax()
    mode2 = round(window_df.loc[vol_idx, "Return"], 2)

    # Mode 3: average return across top 3 volume days
    top_vol = window_df.sort_values("Volume", ascending=False).head(3)
    mode3 = round(top_vol["Return"].mean(), 2)

    return mode1, mode2, mode3


import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


def finbert_sentiment(text, clf, max_chunk_chars=1500, stride_chars=250):
    """
    Compute FinBERT sentiment on long MD&A text by chunking + weighted aggregation.

    Notes:
        - Uses overlap to reduce boundary effects.
        - Removes common boilerplate that dilutes signal.
        - Filters junky chunks (tables / numeric sludge / blobs).
        - Aggregates with length-weights and returns a scalar finbert_score = pos - neg.

    Args:
        text (str): Full MD&A text.
        clf: HF pipeline for ProsusAI/finbert with return_all_scores=True.
        max_chunk_chars (int): Chunk size in characters (proxy for tokens).
        stride_chars (int): Overlap between chunks (chars).

    Returns:
        dict with:
            finbert_pos, finbert_neg, finbert_neu, finbert_score,
            finbert_conf, finbert_chunks, finbert_effective_chunks
        or None if text is empty/unusable.
    """

    def normalize(s):
        return re.sub(r"\s+", " ", (s or "")).strip()

    def remove_boilerplate(s):
        patterns = [
            r"forward[-\s]looking statements.*?(?=\bitem\s*\d\b|\Z)",
            r"cautionary statement.*?(?=\bitem\s*\d\b|\Z)",
            r"safe harbor.*?(?=\bitem\s*\d\b|\Z)",
        ]
        out = s
        for pat in patterns:
            out = re.sub(pat, " ", out, flags=re.IGNORECASE | re.DOTALL)
        return normalize(out)

    def split_chunks(s, size, stride):
        chunks = []
        n = len(s)
        i = 0
        step = max(1, size - stride)
        while i < n:
            chunk = s[i : i + size].strip()
            if chunk:
                chunks.append(chunk)
            if i + size >= n:
                break
            i += step
        return chunks

    def is_good_chunk(s):
        # Keep prose, drop numeric sludge / page artifacts / blobs
        t = normalize(s)
        if len(t) < 80:
            return False

        letters = sum(c.isalpha() for c in t)
        if letters / max(len(t), 1) < 0.15:
            return False

        # Drop base64-ish long runs
        compact = t.replace(" ", "")
        if re.match(r"^[A-Za-z0-9+/=]{250,}$", compact):
            return False

        return True

    text = remove_boilerplate(text)
    if not text:
        return None

    chunks = split_chunks(text, max_chunk_chars, stride_chars)
    if not chunks:
        return None

    chunks = [c for c in chunks if is_good_chunk(c)]
    if not chunks:
        return None

    outputs = clf(chunks)

    pos_list, neg_list, neu_list, w_list, conf_list = [], [], [], [], []
    for chunk, out in zip(chunks, outputs):
        probs = {d["label"].lower(): float(d["score"]) for d in out}
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        neu = probs.get("neutral", 0.0)

        w = max(len(chunk), 1)
        pos_list.append(pos)
        neg_list.append(neg)
        neu_list.append(neu)
        w_list.append(w)
        conf_list.append(max(pos, neg, neu))

    w = np.array(w_list, dtype=float)

    finbert_pos = float(np.average(pos_list, weights=w))
    finbert_neg = float(np.average(neg_list, weights=w))
    finbert_neu = float(np.average(neu_list, weights=w))
    finbert_score = finbert_pos - finbert_neg
    finbert_conf = float(np.average(conf_list, weights=w))

    # Effective number of chunks (like ESS) so you can debug “one chunk dominated”
    w_norm = w / w.sum()
    finbert_effective_chunks = float(1.0 / np.sum(w_norm**2))

    return {
        "finbert_pos": finbert_pos,
        "finbert_neg": finbert_neg,
        "finbert_neu": finbert_neu,
        "finbert_score": finbert_score,
        "finbert_conf": finbert_conf,
        "finbert_chunks": int(len(chunks)),
        "finbert_effective_chunks": finbert_effective_chunks,
    }


def extract_sentiment_and_reaction_returns(
    mdna_dir="data/mdna",
    price_dir="data/price_data",
    output_csv="data/mdna_sentiment_scores.csv",
    max_chunk_chars=1500,
    stride_chars=250,
):
    """
    Analyze MD&A sentiment (FinBERT) and reaction returns for tickers and filing dates.

    Args:
        mdna_dir: Dir containing subfolders with text files for filings
        price_dir: Dir containing daily price CSV files named '{ticker}_daily.csv'
        output_csv: File path where combined sentiment and return scores CSV is saved
        max_chunk_chars: FinBERT chunk size in characters
        stride_chars: FinBERT chunk overlap in characters

    Returns:
        None
    """

    clf = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        return_all_scores=True,
        truncation=True,
    )

    results = []

    for ticker in os.listdir(mdna_dir):
        ticker_dir = os.path.join(mdna_dir, ticker)
        if not os.path.isdir(ticker_dir):
            continue

        for filename in tqdm(sorted(os.listdir(ticker_dir)), desc=f"processing {ticker}"):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(ticker_dir, filename)

            # Read with encoding fallback (filings can be nasty)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin1", errors="ignore") as f:
                    text = f.read()

            if not text.strip():
                continue

            sent = finbert_sentiment(
                text,
                clf,
                max_chunk_chars=max_chunk_chars,
                stride_chars=stride_chars,
            )
            if sent is None:
                continue

            date_str = filename.replace(".txt", "")

            r1, r2, r3 = get_earnings_reaction_returns(
                ticker, date_str, price_dir=price_dir
            )

            results.append(
                {
                    "ticker": ticker,
                    "filing_date": date_str,
                    "return_mode1_signed": r1,
                    "return_mode2_volume_spike": r2,
                    "return_mode3_top3_vol_avg": r3,
                    **sent,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
