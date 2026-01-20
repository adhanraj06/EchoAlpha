# Earnings MD&A Sentiment-Based Contrarian Trading Signal (EchoAlpha)

## Overview
EchoAlpha is a quantitative research project that investigates whether qualitative managerial language in quarterly SEC filings contains incremental information not immediately reflected in equity prices. Specifically, the system analyzes the Management Discussion & Analysis (MD&A) sections of 10-Q filings, extracts sentiment signals, and compares them against short-horizon earnings reaction returns to identify temporary mispricings. The core strategy is contrarian in nature: when management tone and immediate market reaction diverge, the model takes positions betting on mean reversion as narrative information is gradually incorporated into prices. This project is designed as a research-grade pipeline, emphasizing robustness, interpretability, and reproducibility over pure return maximization.

## Research Hypothesis
Markets often overweight structured earnings metrics and short-term flows while underweighting nuanced narrative disclosures. EchoAlpha tests whether discrepancies between MD&A sentiment (management’s qualitative assessment) and immediate price reaction (post-earnings returns) represent exploitable inefficiencies that correct over multi-week horizons.

## Core Modules
- **fetch_ticker_data.py**: Downloads historical adjusted price data and sector metadata
- **analyze_ticker.py**: Performs exploratory price analysis and diagnostics
- **mdna_extractor.py**: Scrapes and extracts MD&A sections from SEC 10-Q filings
- **mdna_sentiment_and_returns.py**: Computes sentiment scores and earnings reaction returns
- **backtest_runner.py**: Constructs signals, simulates portfolios, and evaluates performance
- **config_study.py**: Conducts systematic parameter studies to assess signal stability and robustness

## Setup

Launch venv:

```bash
make install
```

or manually ... 

```bash
python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r src/requirements.txt
```

# How to Run

Run the main pipeline with:

```bash
make run
```

or manually ...

```bash
python main.py
```

# How to Clean

Clean the entire directory (removes all generated files and cached data):
```bash
make clean
```

Clean only generated data (keeps cached files):
```bash
make clean-data
```

# How to Read Documentation

Generate documentation:
```bash
make docs
```

Then open:
```bash
docs/src/index.html
```


## Important Note

The end-to-end pipeline can be computationally and data intensive. The primary bottlenecks are SEC filing ingestion and large-scale simulation, especially SEC 10-Q scraping + MD&A extraction (network + parsing across many filings) and null / randomized backtest pool generation (Monte Carlo-style baselines over many runs). For reference, in a representative run (11 tickers), MD&A extraction took ~9 minutes and the null backtest pool took ~17 minutes, while the remaining steps completed in seconds to a few minutes.
