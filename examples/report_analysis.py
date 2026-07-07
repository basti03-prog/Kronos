# -*- coding: utf-8 -*-
"""
report_analysis.py

Technical-analysis helpers used by investment_report.py: EMA/RSI/MACD,
historical volatility, support/resistance clustering, trend narrative, and
a transparent, weighted bullish/bearish composite score. Pure pandas/numpy,
no model dependency, so it can be tested independently of Kronos.
"""

import numpy as np
import pandas as pd

ANNUALIZATION_FACTORS = {
    "1d": 252, "5d": 52, "1wk": 52,
    "1h": 252 * 7, "30m": 252 * 14, "15m": 252 * 28, "5m": 252 * 84, "1m": 252 * 420,
}


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def historical_volatility(close: pd.Series, interval: str) -> float:
    """Annualized volatility (%) from log returns."""
    log_ret = np.log(close / close.shift(1)).dropna()
    factor = ANNUALIZATION_FACTORS.get(interval, 252)
    return float(log_ret.std() * np.sqrt(factor) * 100.0)


def _cluster_levels(values, tolerance_pct: float):
    """Greedily merges nearby pivot prices into (level, touch_count) clusters."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0], 1]]
    for v in values[1:]:
        level = clusters[-1][0] / clusters[-1][1]
        if abs(v - level) <= tolerance_pct * level:
            clusters[-1][0] += v
            clusters[-1][1] += 1
        else:
            clusters.append([v, 1])
    return [(total / count, count) for total, count in clusters]


def find_support_resistance(df: pd.DataFrame, window: int = 5, tolerance_pct: float = 0.015,
                             num_levels: int = 2):
    """
    Finds swing-high/low pivots in df (columns: high, low, close) and clusters
    them into the nearest support levels (below current price) and resistance
    levels (above current price). Returns (support, resistance), each a list
    of (price, touch_count) sorted by proximity to the current price.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    pivot_highs, pivot_lows = [], []
    for i in range(window, n - window):
        seg_h = highs[i - window:i + window + 1]
        if highs[i] == seg_h.max():
            pivot_highs.append(float(highs[i]))
        seg_l = lows[i - window:i + window + 1]
        if lows[i] == seg_l.min():
            pivot_lows.append(float(lows[i]))

    resistance_clusters = _cluster_levels(pivot_highs, tolerance_pct)
    support_clusters = _cluster_levels(pivot_lows, tolerance_pct)

    current_price = float(df["close"].iloc[-1])
    resistance = sorted([c for c in resistance_clusters if c[0] > current_price],
                         key=lambda c: c[0])[:num_levels]
    support = sorted([c for c in support_clusters if c[0] < current_price],
                      key=lambda c: -c[0])[:num_levels]
    return support, resistance


def describe_trend(last_close: float, ema50: float, ema200: float,
                    forecast_final: float, horizon_label: str) -> str:
    regime = "a Golden Cross (EMA50 above EMA200, long-term uptrend regime)" if ema50 > ema200 \
        else "a Death Cross (EMA50 below EMA200, long-term downtrend regime)"
    position = "above" if last_close > ema50 else "below"
    pct = (forecast_final / last_close - 1.0) * 100.0
    direction = "higher" if pct >= 0 else "lower"
    return (
        f"The market is currently in {regime}. Price is trading {position} its 50-period EMA "
        f"(${ema50:,.2f}) and {'above' if last_close > ema200 else 'below'} its 200-period EMA "
        f"(${ema200:,.2f}). The Kronos ensemble forecast projects the price to move {direction} "
        f"by {pct:+.1f}% over the next {horizon_label}."
    )


def compute_bullish_probability(p_ensemble_up: float, ema50: float, ema200: float,
                                 last_close: float, macd_hist_last: float, rsi_last: float):
    """
    Transparent, weighted composite of four signals, each mapped to [0, 1]
    where 1.0 = maximally bullish:
      - ensemble:  fraction of Monte-Carlo forecast paths ending above the last close
      - trend:     EMA50/EMA200 regime + price position relative to EMA50
      - momentum:  sign of the latest MACD histogram bar
      - rsi:       mean-reversion tilt (oversold -> bullish, overbought -> bearish)
    Weights sum to 1.0 and are fixed/documented, not fit to data.
    """
    if ema50 > ema200 and last_close > ema50:
        trend_component = 1.0
    elif ema50 < ema200 and last_close < ema50:
        trend_component = 0.0
    else:
        trend_component = 0.5

    if macd_hist_last > 0:
        momentum_component = 1.0
    elif macd_hist_last < 0:
        momentum_component = 0.0
    else:
        momentum_component = 0.5

    rsi_component = float(np.clip((70.0 - rsi_last) / 40.0, 0.0, 1.0))

    weights = {"ensemble": 0.50, "trend": 0.20, "momentum": 0.15, "rsi": 0.15}
    components = {
        "Model ensemble (P[price up])": p_ensemble_up,
        "Trend (EMA50/EMA200)": trend_component,
        "Momentum (MACD histogram)": momentum_component,
        "RSI mean-reversion": rsi_component,
    }
    final = sum(weights[k] * v for k, v in zip(
        ["ensemble", "trend", "momentum", "rsi"], components.values()))

    if final >= 0.6:
        verdict = "BULLISH"
    elif final <= 0.4:
        verdict = "BEARISH"
    else:
        verdict = "NEUTRAL / MIXED"

    return final, verdict, components, weights
