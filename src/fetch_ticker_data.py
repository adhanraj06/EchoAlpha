import yfinance as yf
import pandas as pd
import json
import os


def download_price_data(tickers, start_date, data_dir="data/price_data"):
    """Download price data for a list of tickers starting from a given date.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date for historical data in 'YYYY-MM-DD' format.
        data_dir (str, optional): Directory path to save the downloaded CSV files.

    Returns:
        pandas.DataFrame: Raw price data for all tickers with prices grouped by ticker.
    """

    os.makedirs(data_dir, exist_ok=True)
    data = yf.download(tickers, start=start_date, group_by="ticker", auto_adjust=True)

    # For each ticker, check for NaN's (e.g., pre-IPO) and save price data in CSV
    for ticker in tickers:
        if data[ticker]["Close"].isna().any():
            print(f"NaN Found in {ticker}")

        if data[ticker].empty:
            print(f"Empty Data for {ticker}")
        prices = data[ticker].copy()
        prices["Return"] = prices["Close"].pct_change().fillna(0) * 100

        try:
            prices.to_csv(f"{data_dir}/{ticker}_daily.csv")
        except Exception as error:
            print(f"Error Saving {ticker} Data to CSV: {error}")
    return data


def get_ticker_info(ticker):
    """Fetch sector and market cap information for a single ticker via yfinance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Dictionary with keys:
            - "Ticker": ticker symbol
            - "Sector": sector name or "N/A" if unavailable
            - "MarketCap": market capitalization or None if unavailable
    """

    try:
        info = yf.Ticker(ticker).info
        return {
            "Ticker": ticker,
            "Sector": info.get("sector", "N/A"),
            "MarketCap": info.get("marketCap", None),
        }
    except Exception as error:
        print(f"Error Fetching Info for {ticker}: {error}")
        return {"Ticker": ticker, "Sector": "N/A", "MarketCap": None}


def fetch_and_save_sector_info(tickers, json_path="data/dynamic_sector_map.json"):
    """Fetch sector information for a list of tickers, group tickers by sector,
    and save the sector-to-ticker mapping as a JSON file.

    Args:
        tickers (list): List of stock ticker symbols.
        json_path (str, optional): File path to save the JSON sector map.

    Returns:
        dict: Dictionary mapping sector names to lists of ticker symbols.
    """

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    info_list = [get_ticker_info(ticker) for ticker in tickers]
    info_df = pd.DataFrame(info_list)
    dynamic_sector_map = info_df.groupby("Sector")["Ticker"].apply(list).to_dict()

    try:
        with open(json_path, "w") as dsm_file:
            json.dump(dynamic_sector_map, dsm_file, indent=4)
    except Exception as error:
        print(f"Error Saving Sector Map to JSON: {error}")
    return dynamic_sector_map
