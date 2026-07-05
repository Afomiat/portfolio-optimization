import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.models import create_sequences, build_lstm, portfolio_daily_returns


def test_create_sequences_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        create_sequences(np.array([]).reshape(0, 1), window_size=60)


def test_create_sequences_too_short_raises():
    data = np.random.rand(10, 1)
    with pytest.raises(ValueError, match="not more than window_size"):
        create_sequences(data, window_size=60)


def test_create_sequences_success():
    data = np.random.rand(100, 1)
    X, y = create_sequences(data, window_size=60)
    assert X.shape == (40, 60)
    assert y.shape == (40,)


def test_build_lstm_invalid_window_raises():
    with pytest.raises(ValueError, match="positive"):
        build_lstm(window_size=0)


def test_build_lstm_success():
    model = build_lstm(window_size=60)
    assert model.input_shape == (None, 60, 1)


def test_portfolio_daily_returns_missing_asset_raises():
    df = pd.DataFrame({"TSLA": [0.01, 0.02], "BND": [0.001, 0.002]})
    with pytest.raises(ValueError, match="not found in returns_df"):
        portfolio_daily_returns(df, {"SPY": 1.0})


def test_portfolio_daily_returns_weights_dont_sum_to_one_raises():
    df = pd.DataFrame({"TSLA": [0.01, 0.02], "BND": [0.001, 0.002]})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        portfolio_daily_returns(df, {"TSLA": 0.5, "BND": 0.3})


def test_portfolio_daily_returns_success():
    df = pd.DataFrame({"TSLA": [0.01, 0.02], "BND": [0.001, 0.002]})
    result = portfolio_daily_returns(df, {"TSLA": 0.6, "BND": 0.4})
    expected = df["TSLA"] * 0.6 + df["BND"] * 0.4
    pd.testing.assert_series_equal(result, expected, check_names=False)