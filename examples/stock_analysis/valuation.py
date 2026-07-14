# -*- coding: utf-8 -*-
"""
valuation.py

Intrinsic-value estimates: a simplified 2-stage discounted free-cash-flow
(DCF) model and the classic Benjamin Graham number, both compared against
the current market price to derive a margin of safety.

These are standard, well-documented simplifications of real valuation
models — every assumption (discount rate, growth fade, terminal growth) is
returned alongside the result so the report can show its work rather than
present a bare number.
"""

import math
import numpy as np


def estimate_dcf_fair_value(free_cash_flow, shares_outstanding, net_debt, growth_y1,
                             beta=1.0, years=5, risk_free_rate=0.04, equity_risk_premium=0.055,
                             terminal_growth=0.025, growth_bounds=(-0.10, 0.25)):
    """
    2-stage FCF DCF: FCF growth fades linearly from `growth_y1` (clipped to
    `growth_bounds`) toward `terminal_growth` over `years`, discounted at a
    CAPM-style rate (risk-free + beta x equity risk premium), plus a Gordon
    Growth terminal value. Equity value = enterprise value - net debt.
    Returns None if FCF is missing/non-positive (DCF is not meaningful then).
    """
    if free_cash_flow is None or free_cash_flow <= 0 or not shares_outstanding:
        return None

    growth_y1 = float(np.clip(growth_y1 if growth_y1 is not None else 0.05, *growth_bounds))
    beta = beta if beta is not None else 1.0
    discount_rate = risk_free_rate + beta * equity_risk_premium
    discount_rate = max(discount_rate, terminal_growth + 0.02)

    fcf = free_cash_flow
    pv_sum = 0.0
    last_fcf = fcf
    for yr in range(1, years + 1):
        g_yr = growth_y1 + (terminal_growth - growth_y1) * (yr - 1) / max(years - 1, 1)
        fcf = fcf * (1.0 + g_yr)
        pv_sum += fcf / (1.0 + discount_rate) ** yr
        last_fcf = fcf

    terminal_value = last_fcf * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1.0 + discount_rate) ** years

    enterprise_value = pv_sum + pv_terminal
    equity_value = enterprise_value - (net_debt or 0.0)
    fair_value = equity_value / shares_outstanding

    return {
        "fair_value": fair_value,
        "discount_rate": discount_rate,
        "growth_y1": growth_y1,
        "terminal_growth": terminal_growth,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "years": years,
    }


def graham_number(trailing_eps, book_value_per_share):
    """Benjamin Graham's intrinsic-value formula: sqrt(22.5 x EPS x book value/share).
    22.5 = a P/E cap of 15 x a P/B cap of 1.5. Undefined (returns None) if EPS or
    book value is non-positive, since the formula assumes both are."""
    if trailing_eps is None or book_value_per_share is None:
        return None
    if trailing_eps <= 0 or book_value_per_share <= 0:
        return None
    return math.sqrt(22.5 * trailing_eps * book_value_per_share)


def margin_of_safety(current_price, fair_value):
    """(fair value - price) / fair value. Positive = trading below estimated fair value."""
    if fair_value is None or fair_value <= 0 or current_price is None:
        return None
    return (fair_value - current_price) / fair_value


def blended_fair_value(estimates: list):
    """Simple average of the fair-value estimates that are available (None values dropped)."""
    valid = [v for v in estimates if v is not None]
    return float(np.mean(valid)) if valid else None
