import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(actual, predicted):
    """Return MAE, RMSE, MAPE as a dict."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}.")
    if len(actual) == 0:
        raise ValueError("`actual` and `predicted` are empty.")
    if np.any(actual == 0):
        raise ValueError("`actual` contains zero values — MAPE is undefined (division by zero).")

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def performance_metrics(returns, cumulative, risk_free_rate=0.04):
    """Calculate total/annualized return, volatility, Sharpe ratio, and max drawdown for a return series."""
    if returns is None or len(returns) == 0:
        raise ValueError("`returns` is empty.")
    if cumulative is None or len(cumulative) == 0:
        raise ValueError("`cumulative` is empty.")

    total_return = cumulative.iloc[-1] - 1
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_vol = returns.std() * np.sqrt(252)

    if annualized_vol == 0:
        raise ValueError("Annualized volatility is 0 — cannot compute Sharpe ratio (division by zero).")

    sharpe = (annualized_return - risk_free_rate) / annualized_vol

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown
    }