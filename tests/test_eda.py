import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.eda import add_returns_and_volatility, get_outliers, adf_test, risk_metrics


def make_price_df(n=100, seed=42):
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {"Adj Close": prices},
        index=pd.date_range("2024-01-01", periods=n, name="Date")
    )


def test_add_returns_and_volatility_missing_column_raises():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    with pytest.raises(ValueError, match="Adj Close"):
        add_returns_and_volatility(df)


def test_add_returns_and_volatility_too_short_raises():
    df = make_price_df(n=5)
    with pytest.raises(ValueError, match="fewer than the rolling window"):
        add_returns_and_volatility(df, window=21)


def test_add_returns_and_volatility_success():
    df = make_price_df(n=50)
    result = add_returns_and_volatility(df, window=21)
    assert "Daily Return" in result.columns
    assert "Rolling Volatility" in result.columns


def test_get_outliers_missing_column_raises():
    df = pd.DataFrame({"Adj Close": [1, 2, 3]})
    with pytest.raises(ValueError, match="Daily Return"):
        get_outliers(df)


def test_adf_test_too_short_raises():
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="too few"):
        adf_test(series, name="test")


def test_adf_test_returns_expected_keys():
    df = make_price_df(n=200)
    result = adf_test(df["Adj Close"], name="test")
    assert set(result.keys()) == {"name", "adf_statistic", "p_value", "stationary"}


def test_risk_metrics_missing_column_raises():
    df = pd.DataFrame({"Adj Close": [1, 2, 3]})
    with pytest.raises(ValueError, match="Daily Return"):
        risk_metrics(df)


def test_risk_metrics_success():
    df = make_price_df(n=100)
    df = add_returns_and_volatility(df, window=21)
    result = risk_metrics(df)
    assert set(result.keys()) == {"var_95", "annual_return", "annual_volatility", "sharpe_ratio"}