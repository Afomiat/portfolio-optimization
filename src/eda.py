import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def add_returns_and_volatility(df, window=21):
    """Add Daily Return and rolling volatility columns to a price DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, got {type(df)}.")
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column.")
    if len(df) < window:
        raise ValueError(
            f"DataFrame has only {len(df)} rows, fewer than the rolling window ({window}). "
            f"Rolling volatility would be entirely NaN."
        )

    df = df.copy()
    df["Daily Return"] = df["Adj Close"].pct_change()
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=window).std()
    return df


def get_outliers(df, threshold=3):
    """Return rows where Daily Return is more than `threshold` std devs from the mean."""
    if "Daily Return" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'Daily Return' column. "
            "Run add_returns_and_volatility() first."
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    returns = df["Daily Return"].dropna()
    if returns.empty:
        raise ValueError("No non-null Daily Return values to evaluate.")

    mean = returns.mean()
    std = returns.std()
    if std == 0:
        raise ValueError("Standard deviation of returns is 0 — cannot detect outliers.")

    outliers = df[(df["Daily Return"] - mean).abs() > threshold * std]
    return outliers[["Daily Return"]].sort_values("Daily Return")


def adf_test(series, name=""):
    """Run Augmented Dickey-Fuller test, return dict with statistic, p-value, and verdict."""
    clean_series = series.dropna()
    if len(clean_series) < 20:
        raise ValueError(
            f"Series '{name}' has only {len(clean_series)} non-null observations — "
            f"too few for a reliable ADF test (need at least ~20)."
        )

    try:
        result = adfuller(clean_series)
    except Exception as e:
        raise RuntimeError(f"ADF test failed for '{name}': {e}") from e

    return {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "stationary": result[1] < 0.05
    }


def risk_metrics(df, risk_free_rate=0.04):
    """Calculate VaR (95%), annualized return/volatility, and Sharpe ratio."""
    if "Daily Return" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'Daily Return' column. "
            "Run add_returns_and_volatility() first."
        )

    returns = df["Daily Return"].dropna()
    if returns.empty:
        raise ValueError("No non-null Daily Return values to evaluate.")

    annual_std = returns.std() * np.sqrt(252)
    if annual_std == 0:
        raise ValueError("Annualized volatility is 0 — cannot compute Sharpe ratio (division by zero).")

    var_95 = np.percentile(returns, 5)
    annual_return = returns.mean() * 252
    sharpe = (annual_return - risk_free_rate) / annual_std

    return {
        "var_95": var_95,
        "annual_return": annual_return,
        "annual_volatility": annual_std,
        "sharpe_ratio": sharpe
    }