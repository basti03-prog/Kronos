# -*- coding: utf-8 -*-
"""
ai_decision.py

The final decision layer: takes the already-computed weighted Investment
Score (from scoring.py, itself a blend of fundamentals/valuation/technical
momentum/risk/sentiment) and turns it into something an investor can act on:

  - a Buy / Hold / Sell probability split (derived from the score via a
    documented triangular membership function, not a black box)
  - a confidence score (how complete the input data was, and how much the
    underlying sub-scores agree with each other)
  - a 5-tier rating: STRONG BUY / BUY / HOLD / REDUCE / SELL
  - three ranked factor lists, each traced back to a real number so nothing
    is asserted without a data point behind it:
      * Reasons to Buy       (bullish factors)
      * Reasons for Caution  (lower-severity bearish factors, weight < 0.70)
      * Key Risks to Monitor (higher-severity/structural bearish factors, weight >= 0.70)

This module does not introduce a second, competing scoring formula --
it explains and packages the one scoring.py already computed.
"""

from collections import namedtuple
import numpy as np

from . import fundamental_analysis as fa

Factor = namedtuple("Factor", ["label", "detail", "direction", "weight"])

RATING_THRESHOLDS = [
    (80, "STRONG BUY"), (65, "BUY"), (45, "HOLD"), (30, "REDUCE"), (-1, "SELL"),
]


def classify_rating(score: float) -> str:
    for threshold, label in RATING_THRESHOLDS:
        if score >= threshold:
            return label
    return "SELL"


def score_to_probabilities(score: float) -> dict:
    """
    Triangular membership functions centered on Buy (100), Hold (50), and
    Sell (0): each leg ramps linearly to 1.0 at its anchor and 0.0 fifty
    points away, then the three are renormalized to sum to 100%. E.g. a
    score of 90 gives Buy 80% / Hold 20% / Sell 0%; a score of 50 gives
    Hold 100%.
    """
    buy = np.clip((score - 50.0) / 50.0, 0.0, 1.0)
    sell = np.clip((50.0 - score) / 50.0, 0.0, 1.0)
    hold = max(0.0, 1.0 - abs(score - 50.0) / 50.0)
    total = buy + hold + sell
    if total <= 0:
        return {"buy": 0.0, "hold": 100.0, "sell": 0.0}
    return {"buy": buy / total * 100.0, "hold": hold / total * 100.0, "sell": sell / total * 100.0}


def compute_confidence(score_breakdown: dict):
    """
    Confidence = 55% data completeness (how many of the 5 sub-scores could
    be computed at all) + 45% signal agreement (how tightly those sub-scores
    cluster -- a stock that's bullish on every dimension is a higher-
    confidence call than one where fundamentals say buy and technicals say
    sell). Returns (confidence_pct, completeness_frac, agreement_frac).
    """
    vals = [v for v in score_breakdown.values() if v is not None]
    completeness = len(vals) / len(score_breakdown) if score_breakdown else 0.0
    if len(vals) >= 2:
        agreement = 1.0 - float(np.clip(np.std(vals) / 40.0, 0.0, 1.0))
    else:
        agreement = 0.5
    confidence = 100.0 * (0.55 * completeness + 0.45 * agreement)
    return confidence, completeness, agreement


def derive_factors(ctx: dict) -> list:
    """Evaluates a fixed set of rules against the already-computed analysis and
    returns every rule that fired as a Factor(label, detail, direction, weight)."""
    fund = ctx["fund"]
    tech = ctx["tech"]
    risk = ctx["risk"]
    news = ctx["news"]
    factors = []

    def add(label, detail, direction, weight):
        factors.append(Factor(label, detail, direction, weight))

    if not fund.get("error"):
        cur = fund.get("currency") or ""

        # --- Growth ---
        rev_g, earn_g = fund.get("revenue_growth_yoy"), fund.get("earnings_growth_yoy")
        if rev_g is not None and earn_g is not None:
            if rev_g > 0.08 and earn_g > 0.08:
                add("Strong revenue and earnings growth",
                    f"Revenue {rev_g * 100:+.1f}% YoY, earnings {earn_g * 100:+.1f}% YoY",
                    "bullish", 0.90)
            elif rev_g < 0 and earn_g < 0:
                add("Declining revenue and earnings",
                    f"Revenue {rev_g * 100:+.1f}% YoY, earnings {earn_g * 100:+.1f}% YoY",
                    "bearish", 0.90)

        # --- Free cash flow ---
        fcf, fcf_margin = fund.get("free_cash_flow"), fund.get("fcf_margin")
        if fcf is not None and fcf > 0 and fcf_margin is not None and fcf_margin > 0.15:
            add("Strong and growing free cash flow",
                f"FCF margin {fcf_margin * 100:.0f}%, FCF {fa.fmt_money(fcf, cur)}", "bullish", 0.80)
        elif fcf is not None and fcf <= 0:
            add("Negative free cash flow", f"FCF {fa.fmt_money(fcf, cur)}", "bearish", 0.85)

        # --- Balance sheet ---
        d2e, cr = fund.get("debt_to_equity"), fund.get("current_ratio")
        if d2e is not None and d2e < 80 and cr is not None and cr > 1.2:
            add("Healthy, low-leverage balance sheet",
                f"Debt/Equity {d2e:.0f}%, current ratio {cr:.2f}", "bullish", 0.70)
        elif d2e is not None and d2e > 150:
            add("Elevated balance-sheet leverage", f"Debt/Equity {d2e:.0f}%", "bearish", 0.75)

        # --- Margins ---
        om, nm = fund.get("operating_margin"), fund.get("profit_margin")
        if om is not None and om > 0.20 and nm is not None and nm > 0.12:
            add("Strong profit margins",
                f"Operating margin {om * 100:.0f}%, net margin {nm * 100:.0f}%", "bullish", 0.65)
        elif om is not None and om < 0.08:
            add("Weak operating margins", f"Operating margin only {om * 100:.0f}%", "bearish", 0.70)

        # --- Margin trend as a competitive-pressure proxy ---
        years = fund.get("yearly_history", [])
        if len(years) >= 3:
            margins = [y["net_income"] / y["revenue"] for y in years if y.get("revenue")]
            if len(margins) >= 3:
                delta = margins[-1] - margins[0]
                if delta <= -0.03:
                    add("Increasing competitive/cost pressure",
                        f"Net margin compressed from {margins[0] * 100:.0f}% to {margins[-1] * 100:.0f}% "
                        f"over the last {len(margins) - 1} fiscal years", "bearish", 0.55)
                elif delta >= 0.03:
                    add("Expanding margins over time",
                        f"Net margin improved from {margins[0] * 100:.0f}% to {margins[-1] * 100:.0f}% "
                        f"over the last {len(margins) - 1} fiscal years", "bullish", 0.45)

        # --- Analyst consensus ---
        target, rec, last_close = fund.get("target_mean_price"), fund.get("recommendation"), ctx["last_close"]
        if target and last_close:
            upside = target / last_close - 1.0
            if upside > 0.10 and rec in ("buy", "strong_buy"):
                add("Positive analyst consensus",
                    f"Mean target {fa.fmt_money(target, cur)} implies {upside * 100:+.0f}% upside, "
                    f"consensus '{rec}'", "bullish", 0.60)
            elif upside < -0.05 or rec in ("sell", "underperform"):
                add("Cautious analyst consensus",
                    f"Mean target {fa.fmt_money(target, cur)} implies {upside * 100:+.0f}% "
                    f"vs. current price, consensus '{rec}'", "bearish", 0.55)

        # --- Valuation ---
        mos = ctx.get("margin_of_safety_dcf")
        if mos is not None and mos > 0.15:
            add("Attractive valuation", f"Trading {mos * 100:.0f}% below estimated DCF fair value",
                "bullish", 0.85)
        elif mos is not None and mos < -0.30:
            add("High valuation risk", f"Trading {abs(mos) * 100:.0f}% above estimated DCF fair value",
                "bearish", 0.85)

        peg = fund.get("peg_ratio")
        if peg is not None and 0 < peg < 1.5:
            add("Reasonable valuation relative to growth", f"PEG ratio {peg:.2f}", "bullish", 0.40)
        elif peg is not None and peg > 2.5:
            add("Stretched valuation relative to growth", f"PEG ratio {peg:.2f}", "bearish", 0.50)

    # --- Prediction (Kronos model forecast) ---
    p_up = ctx.get("p_ensemble_up")
    exp_ret = ctx.get("forecast_expected_return")
    if p_up is not None and exp_ret is not None:
        if p_up >= 0.65:
            add("Kronos model forecasts continued upside",
                f"{p_up * 100:.0f}% of Monte Carlo paths end higher, average projected move "
                f"{exp_ret * 100:+.1f}% over the forecast horizon", "bullish", 0.80)
        elif p_up <= 0.35:
            add("Kronos model forecasts a decline",
                f"Only {p_up * 100:.0f}% of Monte Carlo paths end higher, average projected move "
                f"{exp_ret * 100:+.1f}% over the forecast horizon", "bearish", 0.80)

    # --- Technical (chart indicators only, independent of the model forecast) ---
    if tech.cross_regime == "golden" and ctx["tech_label"] == "BULLISH":
        add("Strong technical uptrend",
            "Golden Cross regime with bullish MACD/RSI momentum", "bullish", 0.65)
    elif tech.cross_regime == "death" and ctx["tech_label"] == "BEARISH":
        add("Weak technical trend",
            "Death Cross regime with bearish MACD/RSI momentum", "bearish", 0.65)

    rsi_last = float(tech.rsi.iloc[-1])
    if rsi_last >= 75:
        add("Overbought conditions", f"RSI {rsi_last:.0f}", "bearish", 0.35)
    elif rsi_last <= 25:
        add("Oversold -- potential entry point", f"RSI {rsi_last:.0f}", "bullish", 0.35)

    # --- Risk ---
    if risk.sharpe is not None and risk.sharpe > 1.0:
        add("Attractive risk-adjusted returns", f"Sharpe ratio {risk.sharpe:.2f}", "bullish", 0.55)
    elif risk.sharpe is not None and risk.sharpe < 0:
        add("Poor risk-adjusted returns", f"Sharpe ratio {risk.sharpe:.2f}", "bearish", 0.55)

    if risk.max_drawdown_pct is not None and risk.max_drawdown_pct < -35:
        add("History of large drawdowns", f"Max drawdown {risk.max_drawdown_pct:.0f}%", "bearish", 0.40)
    if risk.beta is not None and risk.beta > 1.4:
        add("High volatility vs. the market", f"Beta {risk.beta:.2f}", "bearish", 0.30)

    # --- News sentiment ---
    if news["items"]:
        if news["overall_label"] == "positive":
            add("Positive news sentiment",
                f"{news['counts']['positive']} positive vs {news['counts']['negative']} negative "
                f"recent headlines", "bullish", 0.45)
        elif news["overall_label"] == "negative":
            add("Negative news sentiment",
                f"{news['counts']['negative']} negative vs {news['counts']['positive']} positive "
                f"recent headlines", "bearish", 0.50)

    return factors


# Bearish factors at or above this weight are structural/high-severity concerns ("Key Risks
# to Monitor"); below it they're softer, more tactical flags ("Reasons for Caution").
KEY_RISK_WEIGHT_THRESHOLD = 0.70


def _build_explanation(ticker, rating, score, probabilities, confidence, reasons_to_buy,
                        key_risks, reasons_for_caution):
    lead = f"{ticker} scores {score:.0f}/100 -> {rating} ({confidence:.0f}% confidence)."
    if reasons_to_buy:
        top = reasons_to_buy[0]
        lead += f" Strongest bullish factor: \"{top.label}\" ({top.detail})."
    top_negative = (key_risks or reasons_for_caution)
    if top_negative:
        top = top_negative[0]
        lead += f" Strongest bearish factor: \"{top.label}\" ({top.detail})."
    lead += (f" Probability split -- Buy {probabilities['buy']:.0f}% / Hold {probabilities['hold']:.0f}% "
             f"/ Sell {probabilities['sell']:.0f}%.")
    return lead


def build_decision(ctx: dict) -> dict:
    score = ctx["investment_score"] if ctx["investment_score"] is not None else 50.0
    probabilities = score_to_probabilities(score)
    confidence, completeness, agreement = compute_confidence(ctx["score_breakdown"])
    rating = classify_rating(score)

    factors = derive_factors(ctx)
    bullish = sorted((f for f in factors if f.direction == "bullish"), key=lambda f: -f.weight)
    bearish = sorted((f for f in factors if f.direction == "bearish"), key=lambda f: -f.weight)
    key_risks = [f for f in bearish if f.weight >= KEY_RISK_WEIGHT_THRESHOLD]
    reasons_for_caution = [f for f in bearish if f.weight < KEY_RISK_WEIGHT_THRESHOLD]
    key_drivers = sorted(factors, key=lambda f: -f.weight)[:4]

    explanation = _build_explanation(ctx["ticker"], rating, score, probabilities, confidence,
                                      bullish, key_risks, reasons_for_caution)

    return {
        "score": score, "rating": rating, "probabilities": probabilities,
        "confidence": confidence, "completeness": completeness, "agreement": agreement,
        "reasons_to_buy": bullish[:5], "reasons_for_caution": reasons_for_caution[:5],
        "key_risks": key_risks[:5], "key_drivers": key_drivers, "explanation": explanation,
    }


def format_terminal_summary(ticker: str, decision: dict) -> str:
    lines = ["=" * 72, f"AI INVESTMENT DECISION -- {ticker}", "=" * 72,
              f"Rating:      {decision['rating']}",
              f"Score:       {decision['score']:.0f}/100",
              f"Confidence:  {decision['confidence']:.0f}%  "
              f"(data completeness {decision['completeness'] * 100:.0f}%, "
              f"signal agreement {decision['agreement'] * 100:.0f}%)"]
    p = decision["probabilities"]
    lines.append(f"Buy / Hold / Sell probability: {p['buy']:.0f}% / {p['hold']:.0f}% / {p['sell']:.0f}%")

    for title, key, marker in [("Reasons to Buy", "reasons_to_buy", "+"),
                                ("Reasons for Caution", "reasons_for_caution", "~"),
                                ("Key Risks to Monitor", "key_risks", "!")]:
        lines.append("")
        lines.append(f"{title}:")
        for f in decision[key]:
            lines.append(f"  {marker} {f.label} -- {f.detail}")
        if not decision[key]:
            lines.append("  (none identified)")

    lines.append("")
    lines.append(decision["explanation"])
    lines.append("=" * 72)
    return "\n".join(lines)
