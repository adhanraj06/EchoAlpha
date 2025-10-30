# Earnings MD&A Sentiment-Based Contrarian Trading Signal (EchoAlpha)

## Overview
This project develops a contrarian trading strategy that leverages the qualitative tone of the Management Discussion & Analysis (MD&A) section in quarterly earnings filings (10-Qs). It uses sentiment analysis to quantify management’s narrative tone and compares it with the stock’s price reaction around earnings announcements. The strategy generates contrarian trade signals by betting against the initial market move when sentiment and price reaction diverge, aiming to capture alpha as qualitative insights get priced in over time.

## Core Modules
- **fetch_ticker_data.py**: Downloads historical price and sector data.
- **analyze_ticker.py**: Runs price data analysis and visualizations.
- **mdna_extractor.py**: Extracts MD&A text from SEC filings.
- **mdna_sentiment_and_returns.py**: Computes sentiment scores and earnings reaction returns.
- **backtest_runner.py**: Runs backtests of the trading strategy and visualizes results.

## Setup

Launch venv:
```bash
python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r src/requirements.txt
```

# How to Run

Run the main pipeline with:

```bash
python main.py
```
This will:

- Download price and sector data
- Extract MD&A sections from SEC 10-Q filings (this step is exceptionally resource intensive)
- Compute sentiment scores and earnings reaction returns
- Run backtests and generate performance visualizations

## Important Note

The data download and processing steps in the Notebook and main.py are very computationally and data intensive, especially:

```python
download_and_extract_mdna(tickers, filing_type="10-Q", after="2013-01-01")
extract_sentiment_and_reaction_returns()
```

## Documentation

For quick access to project documentation, run:

```bash
docs/src/index.html
```

