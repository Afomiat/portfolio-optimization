import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.evaluate import evaluate_forecast, performance_metrics


def test_evaluate_forecast_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_forecast([1, 2, 3], [1, 2])


def test_evaluate_forecast_zero_actual_raises():
    with pytest.raises(ValueError, match="MAPE is undefined"):
        evaluate_forecast([0, 1, 2], [1, 1, 2])


def test_evaluate_forecast_success():
    actual = [100, 200, 300]
    predicted = [110, 190, 290]
    result = evaluate_forecast(actual, predicted)
    assert set(result.keys()) == {"mae", "rmse", "mape"}
    assert result["mae"] == pytest.approx(10.0)


def test_performance_metrics_empty_returns_raises():
    with pytest.raises(ValueError, match="empty"):
        performance_metrics(pd.Series([], dtype=float), pd.Series([1.0]))


def test_performance_metrics_zero_volatility_raises():
    returns = pd.Series([0.0, 0.0, 0.0])
    cumulative = pd.Series([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="division by zero"):
        performance_metrics(returns, cumulative)


def test_performance_metrics_success():
    returns = pd.Series([0.01, -0.005, 0.02, 0.01])
    cumulative = (1 + returns).cumprod()
    result = performance_metrics(returns, cumulative)
    assert set(result.keys()) == {
        "total_return", "annualized_return", "annualized_volatility",
        "sharpe_ratio", "max_drawdown"
    }