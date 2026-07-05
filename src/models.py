import numpy as np
import pmdarima as pm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def train_arima(train_series, seasonal=False):
    """Fit auto_arima on a training series, return the fitted model."""
    if train_series is None or len(train_series) < 30:
        raise ValueError(
            f"train_series must have at least 30 observations, got "
            f"{0 if train_series is None else len(train_series)}."
        )
    if train_series.isnull().any():
        raise ValueError("train_series contains NaN values — clean the series before fitting ARIMA.")

    try:
        model = pm.auto_arima(
            train_series,
            start_p=0, start_q=0, max_p=5, max_q=5,
            d=None, seasonal=seasonal,
            trace=False, error_action="ignore",
            suppress_warnings=True, stepwise=True
        )
    except Exception as e:
        raise RuntimeError(f"auto_arima fitting failed: {e}") from e

    return model


def create_sequences(data, window_size=60):
    """Turn a scaled 1D array into (X, y) sliding-window sequences."""
    if data is None or len(data) == 0:
        raise ValueError("`data` is empty — nothing to create sequences from.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if len(data) <= window_size:
        raise ValueError(
            f"`data` has only {len(data)} rows, which is not more than window_size "
            f"({window_size}). No sequences can be created — need at least window_size + 1 rows."
        )

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm(window_size=60):
    """Build and compile the standard LSTM architecture used in this project."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")

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
    if last_sequence is None or len(last_sequence) != window_size:
        raise ValueError(
            f"last_sequence must have exactly window_size ({window_size}) rows, "
            f"got {0 if last_sequence is None else len(last_sequence)}."
        )
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")

    forecast_scaled = []
    current_seq = last_sequence.copy()

    for step in range(n_steps):
        try:
            pred = model.predict(current_seq.reshape(1, window_size, 1), verbose=0)
        except Exception as e:
            raise RuntimeError(f"LSTM prediction failed at step {step}: {e}") from e
        forecast_scaled.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred, axis=0)

    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    return scaler.inverse_transform(forecast_scaled)


def portfolio_daily_returns(returns_df, weights):
    """Compute weighted daily portfolio returns from a DataFrame of asset returns and a weights dict."""
    if not weights:
        raise ValueError("`weights` dict is empty.")

    missing = [asset for asset in weights if asset not in returns_df.columns]
    if missing:
        raise ValueError(f"Assets in weights not found in returns_df columns: {missing}")

    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-4):
        raise ValueError(f"Weights must sum to 1.0, got {total_weight:.4f}.")

    return sum(returns_df[asset] * weight for asset, weight in weights.items())