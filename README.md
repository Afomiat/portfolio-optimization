# Time Series Forecasting for Portfolio Management Optimization

A GMF Investments-style analysis pipeline that forecasts Tesla (TSLA) stock behavior 
and uses it, alongside historical data for BND and SPY, to construct and backtest an 
optimized investment portfolio using Modern Portfolio Theory.

## Project Overview

This project walks through five stages:
1. **EDA** — clean and explore 11+ years of TSLA/BND/SPY price data
2. **Modeling** — build and compare ARIMA and LSTM forecasting models for TSLA
3. **Forecasting** — generate a 12-month forward TSLA forecast with confidence intervals
4. **Portfolio Optimization** — use the forecast + historical data to find the optimal 
   TSLA/BND/SPY allocation via the Efficient Frontier
5. **Backtesting** — validate the recommended portfolio against a simple 60/40 SPY/BND 
   benchmark over an 18-month out-of-sample period

## Project Structure
portfolio-optimization/
├── data/
│   └── processed/          # Cleaned CSVs for TSLA, BND, SPY (generated, not committed raw)
├── notebooks/
│   ├── 1.0-eda.ipynb                    # Task 1: Data exploration, stationarity, risk metrics
│   ├── 2.0-modeling.ipynb               # Task 2: ARIMA vs LSTM training & comparison
│   ├── 3.0-forecast.ipynb               # Task 3: 12-month recursive LSTM forecast
│   ├── 4.0-portfolio-optimization.ipynb # Task 4: Efficient Frontier & optimal weights
│   └── 5.0-backtesting.ipynb            # Task 5: Strategy vs benchmark backtest
├── src/
│   ├── data_loader.py       # Fetch from yfinance, save/load processed CSVs
│   ├── eda.py                # Returns, volatility, outliers, ADF test, risk metrics
│   ├── models.py             # ARIMA training, LSTM architecture, recursive forecasting
│   └── evaluate.py           # MAE/RMSE/MAPE, backtest performance metrics
├── tests/                     # Unit tests (WIP)
├── scripts/
├── requirements.txt
└── README.md

## Setup

```bash
git clone <repo-url>
cd portfolio-optimization
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks in order (1 → 5) from the `notebooks/` folder — each depends on outputs 
from the previous stage (processed data, trained model assumptions, etc.).

## Data

Historical daily price data for TSLA, BND, and SPY, sourced from Yahoo Finance via 
`yfinance`, covering January 1, 2015 – June 30, 2026.

| Asset | Ticker | Role |
|---|---|---|
| Tesla | TSLA | High-growth, high-volatility |
| Vanguard Total Bond Market ETF | BND | Low-risk stability |
| S&P 500 ETF | SPY | Moderate-risk, diversified |

## Key Findings

- **Stationarity:** raw prices are non-stationary (ADF p > 0.7 for all 3 assets); daily 
  returns are stationary (p ≈ 0.0) — confirms `d=1` for ARIMA.
- **ARIMA:** `auto_arima` selected ARIMA(0,1,0) — a random walk — for TSLA, consistent 
  with the Efficient Market Hypothesis (no strong linear autoregressive structure in 
  historical prices alone).
- **LSTM vs ARIMA:** LSTM substantially outperformed ARIMA on one-step-ahead test 
  metrics (MAPE ~4% vs ~17%), but this isn't a fully fair comparison — LSTM's test 
  evaluation used real historical context each day, while ARIMA forecast blind from a 
  single starting point.
- **12-month forecast:** the recursive LSTM forecast (no real data to anchor on) 
  converges toward a learned equilibrium rather than genuinely extrapolating trend, 
  implying an annualized TSLA return of **-43.81%** — treated with explicit skepticism 
  in the portfolio and backtest analysis given this known model limitation.
- **Optimal portfolio:** given the above, both the Max Sharpe (100% SPY) and Min 
  Volatility (~94.5% BND / 5.5% SPY) portfolios exclude TSLA entirely — a direct, 
  mathematically forced consequence of the forecast input, not a general claim about 
  Tesla's fundamentals.
- **Backtest:** the recommended 100% SPY strategy outperformed a static 60/40 SPY/BND 
  benchmark on raw return (28.91% vs 20.88% total return) but had a slightly *worse* 
  Sharpe ratio (0.827 vs 0.870) and a deeper max drawdown (-18.76% vs -11.29%) — more 
  return, but not fully compensated risk-adjusted, and a rougher ride.

## Limitations

- Recursive multi-step LSTM forecasts are known to degrade/flatten over long horizons; 
  the -43.81% TSLA return estimate should not be treated as a confident directional call.
- Backtest uses fixed initial weights (no rebalancing) and ignores transaction costs.
- Single backtest window (Jan 2025 – Jun 2026); results may not generalize to other 
  market regimes.

