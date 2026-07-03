import numpy as np
import pmdarima as pm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def train_arima(train_series, seasonal=False):
    """Fit auto_arima on a training series, return the fitted model."""
    model = pm.auto_arima(
        train_series,
        start_p=0, start_q=0, max_p=5, max_q=5,
        d=None, seasonal=seasonal,
        trace=False, error_action="ignore",
        suppress_warnings=True, stepwise=True
    )
    return model


def create_sequences(data, window_size=60):
    """Turn a scaled 1D array into (X, y) sliding-window sequences."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm(window_size=60):
    """Build and compile the standard LSTM architecture used in this project."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def recursive_forecast(model, last_sequence, n_steps, scaler, window_size=60):
    """Iteratively predict n_steps forward, feeding each prediction back as input."""
    forecast_scaled = []
    current_seq = last_sequence.copy()

    for _ in range(n_steps):
        pred = model.predict(current_seq.reshape(1, window_size, 1), verbose=0)
        forecast_scaled.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred, axis=0)

    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    return scaler.inverse_transform(forecast_scaled)

def portfolio_daily_returns(returns_df, weights):
    """Compute weighted daily portfolio returns from a DataFrame of asset returns and a weights dict."""
    return sum(returns_df[asset] * weight for asset, weight in weights.items())