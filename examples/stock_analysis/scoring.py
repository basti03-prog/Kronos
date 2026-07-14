# -*- coding: utf-8 -*-
"""
scoring.py

Combines every analysis module's output into a single weighted Investment
Score (0-100) across six categories: Prediction (the Kronos Monte-Carlo
forecast), Fundamentals, Technicals (chart indicators), Valuation, Risk, and
News. Every weight is a fixed, documented constant (not fit to data) so the
score is auditable rather than a black box.

Prediction and Technicals are deliberately separate pillars: Prediction is
purely the model's forecast (P[price up] across the ensemble); Technicals is
purely chart-derived (EMA trend, MACD momentum, RSI). Keeping them apart
means the final score shows whether the model and the tape actually agree.
"""

import numpy as np


def prediction_score(p_ensemble_up: float):
    """The Kronos ensemble's own signal: the fraction of Monte-Carlo forecast
    paths that end above the last close, rescaled to [0, 100]."""
    score = float(np.clip(p_ensemble_up, 0.0, 1.0) * 100.0)
    if score >= 60:
        label = "BULLISH"
    elif score <= 40:
        label = "BEARISH"
    else:
        label = "NEUTRAL / MIXED"
    return score, label


def technical_score(ema50: float, ema200: float, last_close: float, macd_hist_last: float,
                     rsi_last: float):
    """
    Pure chart/indicator composite (no model forecast involved), each mapped
    to [0, 100] where 100 = maximally bullish:
      - trend:     EMA50/EMA200 regime (Golden/Death Cross) + price vs EMA50 (40%)
      - momentum:  sign of the latest MACD histogram bar (30%)
      - rsi:       mean-reversion tilt (oversold -> bullish, overbought -> bearish) (30%)
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

    weights = {"trend": 0.40, "momentum": 0.30, "rsi": 0.30}
    components = {
        "Trend (EMA50/EMA200)": trend_component,
        "Momentum (MACD histogram)": momentum_component,
        "RSI mean-reversion": rsi_component,
    }
    score = sum(weights[k] * v for k, v in zip(["trend", "momentum", "rsi"], components.values()))

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


# Fixed category weights, sum to 1.0. Fundamentals + Valuation + Prediction carry the most
# weight as the primary quality/value/model-signal drivers; Technicals/Risk/News confirm.
CATEGORY_WEIGHTS = {
    "Prediction (Kronos)": 0.20,
    "Fundamentals": 0.25,
    "Technicals": 0.15,
    "Valuation": 0.20,
    "Risk": 0.10,
    "News": 0.10,
}


def compute_investment_score(pred_score, quality_score, margin_of_safety, tech_score, sharpe,
                              sentiment_compound):
    """
    Weighted blend across six categories (see CATEGORY_WEIGHTS). Missing
    sub-scores are excluded and the remaining weights renormalized, so the
    score degrades gracefully with sparser data.
    """
    sub_scores = {
        "Prediction (Kronos)": pred_score,
        "Fundamentals": quality_score,
        "Valuation": _score_margin_of_safety(margin_of_safety),
        "Technicals": tech_score,
        "Risk": _score_sharpe(sharpe),
        "News": _score_sentiment(sentiment_compound) if sentiment_compound is not None else None,
    }
    weights = dict(CATEGORY_WEIGHTS)

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
