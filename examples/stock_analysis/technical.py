# -*- coding: utf-8 -*-
"""
technical.py

Classic technical-analysis indicators computed with plain pandas/numpy:
EMA, RSI, MACD, Bollinger Bands, ATR, ADX, support/resistance clustering,
and Golden/Death Cross detection. No model dependency.
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


def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth_pct = (upper - lower) / mid * 100.0
    return mid, upper, lower, bandwidth_pct


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing). df needs columns high, low, close."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14):
    """Average Directional Index (Wilder). Returns (adx, plus_di, minus_di)."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr = compute_atr(df, period)
    atr_safe = atr.replace(0.0, np.nan)

    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_safe
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_safe

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


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


def detect_cross(ema_fast: pd.Series, ema_slow: pd.Series):
    """
    Detects the most recent Golden Cross (fast crossing above slow) or Death
    Cross (fast crossing below slow). Returns (regime, bars_since_cross) where
    regime is "golden" or "death"; bars_since_cross is None if no cross exists
    in the provided window.
    """
    diff = (ema_fast - ema_slow).reset_index(drop=True)
    sign = np.sign(diff)
    crosses = np.where(np.diff(sign) != 0)[0]
    regime = "golden" if diff.iloc[-1] > 0 else "death"
    bars_since = int(len(diff) - 1 - crosses[-1]) if len(crosses) else None
    return regime, bars_since


class TechnicalSnapshot:
    """Container for the full set of technical indicators computed over a price history."""

    def __init__(self, df: pd.DataFrame, lookback: int, interval: str, sr_window: int = 5,
                 sr_tolerance: float = 0.015):
        self.interval = interval
        close = df["close"]

        self.ema20_full = compute_ema(close, 20)
        self.ema50_full = compute_ema(close, 50)
        self.ema200_full = compute_ema(close, 200)
        self.rsi_full = compute_rsi(close, 14)
        self.macd_full, self.signal_full, self.hist_full = compute_macd(close)
        self.bb_mid_full, self.bb_upper_full, self.bb_lower_full, self.bb_width_full = \
            compute_bollinger_bands(close, 20, 2.0)
        self.atr_full = compute_atr(df, 14)
        self.adx_full, self.plus_di_full, self.minus_di_full = compute_adx(df, 14)

        tail = lambda s: s.iloc[-lookback:].reset_index(drop=True)  # noqa: E731
        self.ema20 = tail(self.ema20_full)
        self.ema50 = tail(self.ema50_full)
        self.ema200 = tail(self.ema200_full)
        self.rsi = tail(self.rsi_full)
        self.macd = tail(self.macd_full)
        self.signal = tail(self.signal_full)
        self.hist = tail(self.hist_full)
        self.bb_mid = tail(self.bb_mid_full)
        self.bb_upper = tail(self.bb_upper_full)
        self.bb_lower = tail(self.bb_lower_full)
        self.bb_width = tail(self.bb_width_full)
        self.atr = tail(self.atr_full)
        self.adx = tail(self.adx_full)
        self.plus_di = tail(self.plus_di_full)
        self.minus_di = tail(self.minus_di_full)

        self.hist_vol_annualized = historical_volatility(close, interval)
        self.support, self.resistance = find_support_resistance(
            df.iloc[-lookback:], window=sr_window, tolerance_pct=sr_tolerance)
        self.cross_regime, self.bars_since_cross = detect_cross(self.ema50_full, self.ema200_full)

        self.last_close = float(close.iloc[-1])
        self.atr_pct = float(self.atr.iloc[-1] / self.last_close * 100.0)
        self.trend_strength = "strong" if self.adx.iloc[-1] >= 25 else "weak/absent"
