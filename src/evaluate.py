import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(actual, predicted):
    """Return MAE, RMSE, MAPE as a dict."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}