# Algorithmic Trading & Backtesting Engine (Python)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

## Overview
A production-grade, vectorized quantitative trading engine built in Python using **Pandas**, **NumPy**, and **SciPy**. This engine implements a **Mean-Reversion & Moving-Average strategy** benchmarked against 5 years of historical price data for major European equities (Euro Stoxx 50 universe).

## Key Strategy Metrics
- **Annualized Sharpe Ratio**: **1.45**
- **Maximum Drawdown**: **-11.0%**
- **Backtest Horizon**: 5 Years (2019–2024)
- **Execution Mechanism**: Vectorized backtest with a 1-day lag to strictly eliminate **Look-Ahead Bias** and account for execution latency.

---

## Strategy & Mathematical Formulation
The signal generation models statistical price deviations relative to a rolling Gaussian distribution:

$$Z_t = \frac{P_t - \mu_{t, N}}{\sigma_{t, N}}$$

Where:
- $P_t$: Closing price at time $t$
- $\mu_{t, N}$: Rolling mean over lookback window $N = 20$ days
- $\sigma_{t, N}$: Rolling standard deviation over $N = 20$ days

### Execution Rules
- **Long Position ($+1.0$)**: Triggered when $Z_t < -1.5$ (asset statistically undervalued).
- **Short Position ($-1.0$)**: Triggered when $Z_t > +1.5$ (asset statistically overvalued).
- **Exit Condition**: Position closed when $Z_t$ reverts to the mean ($Z_t \approx 0$).

---

## Repository Structure
```text
├── src/
│   ├── strategies.py      # Quantitative signal generation algorithms
│   └── backtester.py      # Vectorized performance & risk evaluation engine
├── notebooks/
│   └── backtest_analysis.ipynb # Parameter optimization & visual analytics
├── tests/
│   └── test_backtester.py # Unit tests verifying Sharpe & Drawdown calculations
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
