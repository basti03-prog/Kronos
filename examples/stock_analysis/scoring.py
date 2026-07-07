# -*- coding: utf-8 -*-
"""
scoring.py

Combines every analysis module's output into two things:
  1. A technical momentum/trend composite (0-100) — the "does the tape agree" signal.
  2. A final Investment Score (0-100) blending fundamentals, valuation, momentum,
     risk, and news sentiment into a single Buy/Hold/Sell recommendation with
     programmatically generated supporting/detracting reasons.

Every weight is a fixed, documented constant (not fit to data) so the score is
auditable rather than a black box.
"""

import numpy as np


def technical_momentum_score(p_ensemble_up: float, ema50: float, ema200: float, last_close: float,
                              macd_hist_last: float, rsi_last: float):
    """
    Transparent, weighted composite of four signals, each mapped to [0, 100]
    where 100 = maximally bullish:
      - ensemble:  % of Monte-Carlo forecast paths ending above the last close
      - trend:     EMA50/EMA200 regime (Golden/Death Cross) + price vs EMA50
      - momentum:  sign of the latest MACD histogram bar
      - rsi:       mean-reversion tilt (oversold -> bullish, overbought -> bearish)
    """
    if ema50 > ema200 and last_close > ema50:
        trend_component = 100.0
    elif ema50 < ema200 and last_close < ema50:
        trend_component = 0.0
    else:
        trend_component = 50.0

    if macd_hist_last > 0:
        momentum_component = 100.0
    elif macd_hist_last < 0:
        momentum_component = 0.0
    else:
        momentum_component = 50.0

    rsi_component = float(np.clip((70.0 - rsi_last) / 40.0, 0.0, 1.0) * 100.0)

    weights = {"ensemble": 0.50, "trend": 0.20, "momentum": 0.15, "rsi": 0.15}
    components = {
        "Model ensemble (P[price up])": p_ensemble_up * 100.0,
        "Trend (EMA50/EMA200)": trend_component,
        "Momentum (MACD histogram)": momentum_component,
        "RSI mean-reversion": rsi_component,
    }
    score = sum(weights[k] * v for k, v in zip(
        ["ensemble", "trend", "momentum", "rsi"], components.values()))

    if score >= 60:
        label = "BULLISH"
    elif score <= 40:
        label = "BEARISH"
    else:
        label = "NEUTRAL / MIXED"

    return score, label, components, weights


def _score_margin_of_safety(mos):
    """MoS >= +30% -> 100 (deeply undervalued), MoS <= -30% -> 0 (deeply overvalued), linear between."""
    if mos is None:
        return None
    return float(np.clip((mos + 0.30) / 0.60, 0.0, 1.0) * 100.0)


def _score_sharpe(sharpe):
    """Sharpe >= 1.5 -> 100, Sharpe <= -0.5 -> 0, linear between."""
    if sharpe is None:
        return None
    return float(np.clip((sharpe + 0.5) / 2.0, 0.0, 1.0) * 100.0)


def _score_sentiment(avg_compound):
    """VADER compound in [-1, 1] -> [0, 100]."""
    return float(np.clip((avg_compound + 1.0) / 2.0, 0.0, 1.0) * 100.0)


def compute_investment_score(quality_score, margin_of_safety, momentum_score, sharpe,
                              sentiment_compound):
    """
    Weighted blend of fundamental quality (30%), valuation/margin-of-safety (25%),
    technical momentum (20%), risk-adjusted return via Sharpe (15%), and news
    sentiment (10%). Missing sub-scores are excluded and the remaining weights
    renormalized, so the score degrades gracefully with sparser data.
    """
    sub_scores = {
        "Fundamental quality": quality_score,
        "Valuation (MoS)": _score_margin_of_safety(margin_of_safety),
        "Technical momentum": momentum_score,
        "Risk-adj. return": _score_sharpe(sharpe),
        "News sentiment": _score_sentiment(sentiment_compound) if sentiment_compound is not None else None,
    }
    weights = {
        "Fundamental quality": 0.30,
        "Valuation (MoS)": 0.25,
        "Technical momentum": 0.20,
        "Risk-adj. return": 0.15,
        "News sentiment": 0.10,
    }

    valid = {k: v for k, v in sub_scores.items() if v is not None}
    if not valid:
        return None, "HOLD", sub_scores, weights, []

    total_weight = sum(weights[k] for k in valid)
    final_score = sum(weights[k] * v for k, v in valid.items()) / total_weight

    if final_score >= 65:
        recommendation = "BUY"
    elif final_score <= 40:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    reasons = []
    for name, val in sorted(valid.items(), key=lambda kv: kv[1], reverse=True):
        direction = "supports" if val >= 55 else ("weighs against" if val <= 45 else "is neutral for")
        reasons.append(f"{name}: {val:.0f}/100 — {direction} the case.")

    return final_score, recommendation, sub_scores, weights, reasons
