import numpy as np
from statsmodels.tsa.stattools import adfuller


def add_returns_and_volatility(df, window=21):
    """Add Daily Return and rolling volatility columns to a price DataFrame."""
    df = df.copy()
    df["Daily Return"] = df["Adj Close"].pct_change()
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=window).std()
    return df


def get_outliers(df, threshold=3):
    """Return rows where Daily Return is more than `threshold` std devs from the mean."""
    mean = df["Daily Return"].mean()
    std = df["Daily Return"].std()
    outliers = df[(df["Daily Return"] - mean).abs() > threshold * std]
    return outliers[["Daily Return"]].sort_values("Daily Return")


def adf_test(series, name=""):
    """Run Augmented Dickey-Fuller test, return dict with statistic, p-value, and verdict."""
    result = adfuller(series.dropna())
    return {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "stationary": result[1] < 0.05
    }


def risk_metrics(df, risk_free_rate=0.04):
    """Calculate VaR (95%), annualized return/volatility, and Sharpe ratio."""
    returns = df["Daily Return"].dropna()
    var_95 = np.percentile(returns, 5)
    annual_return = returns.mean() * 252
    annual_std = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_std
    return {
        "var_95": var_95,
        "annual_return": annual_return,
        "annual_volatility": annual_std,
        "sharpe_ratio": sharpe
    }