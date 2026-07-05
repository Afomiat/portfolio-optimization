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

    Raises:
        ValueError: if tickers is empty or dates are invalid.
        RuntimeError: if yfinance returns no data (bad ticker, network issue, etc.).
    """
    if not tickers:
        raise ValueError("`tickers` must be a non-empty list, e.g. ['TSLA', 'BND', 'SPY'].")

    try:
        pd.Timestamp(start_date)
        pd.Timestamp(end_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid start_date or end_date: {e}") from e

    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date,
                                group_by="ticker", auto_adjust=False)
    except Exception as e:
        raise RuntimeError(f"yfinance download failed for tickers={tickers}: {e}") from e

    if raw_data.empty:
        raise RuntimeError(
            f"yfinance returned no data for tickers={tickers} between "
            f"{start_date} and {end_date}. Check ticker symbols and date range."
        )

    result = {}
    for ticker in tickers:
        try:
            ticker_df = raw_data[ticker].copy()
        except KeyError:
            raise RuntimeError(
                f"No data returned for ticker '{ticker}' — it may be delisted, "
                f"misspelled, or unavailable for this date range."
            )
        if ticker_df.dropna(how="all").empty:
            raise RuntimeError(f"Ticker '{ticker}' returned only empty/NaN rows.")
        result[ticker] = ticker_df

    return result


def save_processed(data_dict, output_dir=None):
    """
    Save each ticker's DataFrame to CSV.

    Raises:
        ValueError: if data_dict is empty or contains non-DataFrame values.
        OSError: if the output directory can't be created or written to.
    """
    if not data_dict:
        raise ValueError("`data_dict` is empty — nothing to save.")

    output_dir = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create output directory '{output_dir}': {e}") from e

    for ticker, df in data_dict.items():
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected a DataFrame for '{ticker}', got {type(df)}.")
        if df.empty:
            raise ValueError(f"DataFrame for '{ticker}' is empty — refusing to save an empty file.")
        try:
            df.to_csv(output_dir / f"{ticker.lower()}.csv")
        except OSError as e:
            raise OSError(f"Failed to write CSV for '{ticker}' to {output_dir}: {e}") from e


def load_processed(ticker, data_dir=None):
    """
    Load a single ticker's processed CSV as a DataFrame.

    Raises:
        FileNotFoundError: if the expected CSV doesn't exist, with a hint to run
            the data-loading step first.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    file_path = data_dir / f"{ticker.lower()}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"No processed data found for '{ticker}' at {file_path}. "
            f"Run fetch_data() + save_processed() first (see notebooks/1.0-eda.ipynb)."
        )

    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    if df.empty:
        raise ValueError(f"Processed file for '{ticker}' at {file_path} is empty.")
    return df