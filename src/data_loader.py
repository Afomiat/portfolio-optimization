import yfinance as yf
import pandas as pd
from pathlib import Path

# Resolve project root relative to this file's location (src/../ = project root)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def fetch_data(tickers, start_date, end_date):
    """
    Fetch historical data for given tickers from yfinance.
    Returns a dict of {ticker: DataFrame} with flat columns.
    """
    raw_data = yf.download(tickers, start=start_date, end=end_date,
                            group_by="ticker", auto_adjust=False)
    return {ticker: raw_data[ticker].copy() for ticker in tickers}


def save_processed(data_dict, output_dir=None):
    """Save each ticker's DataFrame to CSV."""
    output_dir = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
    for ticker, df in data_dict.items():
        df.to_csv(output_dir / f"{ticker.lower()}.csv")


def load_processed(ticker, data_dir=None):
    """Load a single ticker's processed CSV as a DataFrame."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return pd.read_csv(data_dir / f"{ticker.lower()}.csv",
                        index_col="Date", parse_dates=True)