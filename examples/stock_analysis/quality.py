# -*- coding: utf-8 -*-
"""
quality.py

Company-quality narrative: economic moat, market position, growth
opportunities and key risks. Everything here is either (a) factual data
sourced from Yahoo's company profile (sector, industry, business summary),
or (b) a quantitative proxy explicitly derived from the fundamentals/
technicals already computed elsewhere (margin level, ROIC vs. a cost-of-
capital benchmark, leverage, valuation stretch, growth consistency).

This module deliberately does NOT fabricate subjective claims (e.g. "wide
moat from network effects") that would require real qualitative research —
it states what the numbers imply and flags where that stops.
"""


def assess_company_quality(fund: dict, risk, valuation_summary: dict) -> dict:
    moat_signals = []
    growth_signals = []
    risk_signals = []

    gm = fund.get("gross_margin")
    if gm is not None:
        if gm >= 0.50:
            moat_signals.append(
                f"Gross margin of {gm * 100:.0f}% is high, a common quantitative proxy for "
                f"pricing power / limited direct price competition (a possible moat indicator)."
            )
        elif gm < 0.25:
            moat_signals.append(
                f"Gross margin of {gm * 100:.0f}% suggests a commodity-like, price-competitive "
                f"business with limited pricing power."
            )

    roic = fund.get("roic")
    if roic is not None:
        if roic >= 0.15:
            moat_signals.append(
                f"Approximate ROIC of {roic * 100:.0f}% is comfortably above a typical ~8-10% cost "
                f"of capital, consistent with a competitive advantage generating excess returns."
            )
        elif roic < 0.08:
            moat_signals.append(
                f"Approximate ROIC of {roic * 100:.0f}% is near or below a typical cost of capital, "
                f"suggesting limited economic-profit generation at present."
            )

    years = fund.get("yearly_history", [])
    if len(years) >= 3:
        revs = [y["revenue"] for y in years]
        up_years = sum(1 for i in range(1, len(revs)) if revs[i] > revs[i - 1])
        growth_signals.append(
            f"Revenue increased in {up_years}/{len(revs) - 1} of the last {len(revs) - 1} "
            f"fiscal years on record."
        )
    cagr = fund.get("revenue_cagr")
    if cagr is not None:
        growth_signals.append(f"Revenue CAGR over that period: {cagr * 100:+.1f}% per year.")

    eps_t, eps_f = fund.get("trailing_eps"), fund.get("forward_eps")
    if eps_t and eps_f:
        implied = eps_f / eps_t - 1.0
        growth_signals.append(
            f"Analyst consensus implies forward EPS growth of {implied * 100:+.1f}% "
            f"(trailing EPS {eps_t:.2f} -> forward EPS {eps_f:.2f})."
        )

    # Risk signals
    d2e = fund.get("debt_to_equity")
    if d2e is not None and d2e > 150:
        risk_signals.append(
            f"Debt/Equity of {d2e:.0f}% is elevated; balance-sheet risk warrants monitoring, "
            f"especially if rates rise or operating cash flow weakens."
        )
    fcf = fund.get("free_cash_flow")
    if fcf is not None and fcf <= 0:
        risk_signals.append("Free cash flow is currently negative or zero.")
    peg = fund.get("peg_ratio")
    if peg is not None and peg > 2.5:
        risk_signals.append(
            f"PEG ratio of {peg:.1f} suggests the current valuation prices in a lot of future "
            f"growth — a growth disappointment could compress the multiple."
        )
    if risk is not None and risk.max_drawdown_pct is not None and risk.max_drawdown_pct < -40:
        risk_signals.append(
            f"Maximum drawdown over the analysis window reached {risk.max_drawdown_pct:.0f}%, "
            f"indicating meaningful historical volatility/downside risk."
        )
    if risk is not None and risk.beta is not None and risk.beta > 1.4:
        risk_signals.append(
            f"Computed beta of {risk.beta:.2f} indicates materially higher volatility than the "
            f"broad market (S&P 500)."
        )
    mos_dcf = valuation_summary.get("margin_of_safety_dcf")
    if mos_dcf is not None and mos_dcf < -0.30:
        risk_signals.append(
            f"Trading {abs(mos_dcf) * 100:.0f}% above the DCF fair-value estimate — the current "
            f"price already assumes a favorable growth/margin scenario."
        )
    if not risk_signals:
        risk_signals.append(
            "No major red flags in the computed leverage/cash-flow/valuation/drawdown checks above."
        )

    market_position = []
    if fund.get("sector") or fund.get("industry"):
        market_position.append(
            f"Operates in the '{fund.get('industry', 'n/a')}' industry within the "
            f"'{fund.get('sector', 'n/a')}' sector."
        )
    if fund.get("market_cap"):
        market_position.append(f"Market capitalization: {_fmt_money(fund['market_cap'], fund.get('currency', ''))}.")
    if fund.get("employees"):
        market_position.append(f"Full-time employees: {fund['employees']:,}.")

    return {
        "business_summary": fund.get("business_summary"),
        "market_position": market_position,
        "moat_signals": moat_signals or ["Insufficient margin/ROIC data to assess a moat proxy."],
        "growth_opportunities": growth_signals or ["Insufficient growth history available."],
        "key_risks": risk_signals,
        "caveat": (
            "These signals are quantitative proxies derived from public financial-statement "
            "data, not qualitative competitive research. Real moat analysis (brand strength, "
            "switching costs, network effects, regulatory position, customer concentration, "
            "litigation exposure) requires human research beyond what this tool can compute."
        ),
    }


def _fmt_money(value, currency=""):
    if value is None:
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{abs_v / 1e12:.2f}T {currency}".strip()
    if abs_v >= 1e9:
        return f"{abs_v / 1e9:.2f}B {currency}".strip()
    if abs_v >= 1e6:
        return f"{abs_v / 1e6:.2f}M {currency}".strip()
    return f"{abs_v:,.0f} {currency}".strip()
