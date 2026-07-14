# -*- coding: utf-8 -*-
"""
risk.py

Risk metrics computed directly from price history: beta (vs. a benchmark
index), maximum drawdown, Sharpe ratio, and Sortino ratio. Historical
volatility itself lives in technical.py (it doubles as a technical-analysis
input) and is reused here rather than duplicated.
"""

import numpy as np
import pandas as pd

from .technical import ANNUALIZATION_FACTORS


def compute_beta(stock_close: pd.Series, stock_dates: pd.Series, bench_close: pd.Series,
                  bench_dates: pd.Series):
    """Beta = Cov(stock returns, benchmark returns) / Var(benchmark returns), aligned by date."""
    stock_df = pd.DataFrame({"date": stock_dates.values, "stock": stock_close.values})
    bench_df = pd.DataFrame({"date": bench_dates.values, "bench": bench_close.values})
    merged = pd.merge(stock_df, bench_df, on="date", how="inner").sort_values("date")
    if len(merged) < 30:
        return None
    stock_ret = merged["stock"].pct_change().dropna()
    bench_ret = merged["bench"].pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 30:
        return None
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    return float(cov / var) if var else None


def max_drawdown(close: pd.Series, dates: pd.Series):
    """Returns (max_drawdown_pct, peak_date, trough_date) over the given close series."""
    close = close.reset_index(drop=True)
    dates = dates.reset_index(drop=True)
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    trough_pos = int(drawdown.idxmin())
    peak_pos = int(close.iloc[:trough_pos + 1].idxmax())
    return float(drawdown.iloc[trough_pos]) * 100.0, dates.iloc[peak_pos], dates.iloc[trough_pos]


def sharpe_ratio(close: pd.Series, interval: str, risk_free_annual: float = 0.04):
    """Annualized Sharpe ratio from simple period returns."""
    periods_per_year = ANNUALIZATION_FACTORS.get(interval, 252)
    returns = close.pct_change().dropna()
    if returns.std() == 0 or len(returns) < 10:
        return None
    rf_period = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_period
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(close: pd.Series, interval: str, risk_free_annual: float = 0.04):
    """Annualized Sortino ratio (downside deviation only) from simple period returns."""
    periods_per_year = ANNUALIZATION_FACTORS.get(interval, 252)
    returns = close.pct_change().dropna()
    if len(returns) < 10:
        return None
    rf_period = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_period
    downside = excess[excess < 0]
    downside_std = downside.std()
    if not downside_std:
        return None
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


class RiskSnapshot:
    def __init__(self, df: pd.DataFrame, lookback: int, interval: str, benchmark_df: pd.DataFrame = None,
                 risk_free_annual: float = 0.04):
        window = df.iloc[-lookback:]
        self.max_drawdown_pct, self.drawdown_peak_date, self.drawdown_trough_date = \
            max_drawdown(window["close"], window["date"])
        self.sharpe = sharpe_ratio(window["close"], interval, risk_free_annual)
        self.sortino = sortino_ratio(window["close"], interval, risk_free_annual)
        self.beta = (
            compute_beta(window["close"], window["date"], benchmark_df["close"], benchmark_df["date"])
            if benchmark_df is not None else None
        )
        self.risk_free_annual = risk_free_annual
