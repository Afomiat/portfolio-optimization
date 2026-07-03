import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(actual, predicted):
    """Return MAE, RMSE, MAPE as a dict."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def performance_metrics(returns, cumulative, risk_free_rate=0.04):
    """Calculate total/annualized return, volatility, Sharpe ratio, and max drawdown for a return series."""
    total_return = cumulative.iloc[-1] - 1
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_vol = returns.std() * np.sqrt(252)
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