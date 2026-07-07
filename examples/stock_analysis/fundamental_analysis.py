# -*- coding: utf-8 -*-
"""
fundamental_analysis.py

Parses Yahoo Finance's quoteSummary payload into a flat fundamentals dict
(valuation, growth, margins, cash flow, leverage, dividends, company profile)
and computes a transparent, documented 0-100 Quality Score from it.
"""

import datetime
import numpy as np

from .data_provider import _raw


def parse_fundamentals(qs_raw: dict) -> dict:
    """Flattens the modules returned by data_provider.fetch_quote_summary()."""
    if not qs_raw:
        return {"error": "No fundamentals data available for this ticker"}

    fin = qs_raw.get("financialData", {})
    stats = qs_raw.get("defaultKeyStatistics", {})
    summ = qs_raw.get("summaryDetail", {})
    price = qs_raw.get("price", {})
    profile = qs_raw.get("assetProfile", {})

    data = {
        "currency": _raw(price, "currency"),
        "long_name": _raw(price, "longName") or _raw(price, "shortName"),
        "market_cap": _raw(price, "marketCap"),

        "sector": _raw(profile, "sector"),
        "industry": _raw(profile, "industry"),
        "employees": _raw(profile, "fullTimeEmployees"),
        "business_summary": _raw(profile, "longBusinessSummary"),
        "website": _raw(profile, "website"),

        # Valuation
        "trailing_pe": _raw(summ, "trailingPE"),
        "forward_pe": _raw(stats, "forwardPE") or _raw(summ, "forwardPE"),
        "peg_ratio": _raw(stats, "pegRatio"),
        "price_to_book": _raw(stats, "priceToBook"),
        "price_to_sales": _raw(stats, "priceToSalesTrailing12Months"),
        "ev_to_ebitda": _raw(stats, "enterpriseToEbitda"),
        "ev_to_revenue": _raw(stats, "enterpriseToRevenue"),
        "trailing_eps": _raw(stats, "trailingEps"),
        "forward_eps": _raw(stats, "forwardEps"),
        "book_value_per_share": _raw(stats, "bookValue"),
        "shares_outstanding": _raw(stats, "sharesOutstanding"),

        # Dividends
        "dividend_yield": _raw(summ, "dividendYield"),
        "dividend_rate": _raw(summ, "dividendRate"),
        "payout_ratio": _raw(summ, "payoutRatio"),

        "beta": _raw(summ, "beta") or _raw(stats, "beta"),

        # Growth
        "revenue_growth_yoy": _raw(fin, "revenueGrowth"),
        "earnings_growth_yoy": _raw(fin, "earningsGrowth"),
        "revenue_quarterly_growth": _raw(stats, "revenueQuarterlyGrowth"),
        "earnings_quarterly_growth": _raw(stats, "earningsQuarterlyGrowth"),

        # Profitability
        "total_revenue": _raw(fin, "totalRevenue"),
        "ebitda": _raw(fin, "ebitda"),
        "gross_margin": _raw(fin, "grossMargins"),
        "operating_margin": _raw(fin, "operatingMargins"),
        "profit_margin": _raw(fin, "profitMargins"),
        "return_on_equity": _raw(fin, "returnOnEquity"),
        "return_on_assets": _raw(fin, "returnOnAssets"),

        # Cash flow & balance sheet
        "free_cash_flow": _raw(fin, "freeCashflow"),
        "operating_cash_flow": _raw(fin, "operatingCashflow"),
        "total_cash": _raw(fin, "totalCash"),
        "total_debt": _raw(fin, "totalDebt"),
        "debt_to_equity": _raw(fin, "debtToEquity"),
        "current_ratio": _raw(fin, "currentRatio"),
        "quick_ratio": _raw(fin, "quickRatio"),

        # Analyst
        "target_mean_price": _raw(fin, "targetMeanPrice"),
        "recommendation": _raw(fin, "recommendationKey"),
        "num_analyst_opinions": _raw(fin, "numberOfAnalystOpinions"),
    }

    data["net_debt"] = (
        data["total_debt"] - data["total_cash"]
        if data["total_debt"] is not None and data["total_cash"] is not None else None
    )
    data["fcf_margin"] = (
        data["free_cash_flow"] / data["total_revenue"]
        if data["free_cash_flow"] is not None and data["total_revenue"] else None
    )

    # ROIC approximation: no precise EBIT/effective-tax-rate is exposed by this
    # endpoint, so NOPAT is proxied from the operating margin and a flat 21% US
    # statutory tax-rate assumption, and invested capital from book equity
    # (book value/share x shares outstanding) + debt - cash. Documented as an
    # approximation everywhere it's displayed.
    equity_book = (
        data["book_value_per_share"] * data["shares_outstanding"]
        if data["book_value_per_share"] is not None and data["shares_outstanding"] else None
    )
    data["equity_book_value"] = equity_book
    if (equity_book is not None and data["total_debt"] is not None
            and data["total_cash"] is not None and data["operating_margin"] is not None
            and data["total_revenue"]):
        invested_capital = equity_book + data["total_debt"] - data["total_cash"]
        nopat = data["operating_margin"] * data["total_revenue"] * (1 - 0.21)
        # Companies with heavy share buybacks (e.g. Apple) can have book equity that's tiny
        # relative to earnings power, which blows up this book-value-based proxy toward
        # infinity. Require invested capital to be at least 10% of revenue before trusting
        # the ratio, and clip to a sane ceiling either way -- this is a proxy, not GAAP ROIC.
        if invested_capital > 0.10 * data["total_revenue"]:
            data["roic"] = float(np.clip(nopat / invested_capital, -1.0, 0.60))
        else:
            data["roic"] = None
    else:
        data["roic"] = None

    # Multi-year revenue/net-income trend (oldest -> newest); Yahoo's other legacy
    # statement fields (balance sheet, historical cash flow) are no longer populated.
    history = qs_raw.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])
    years = []
    for row in reversed(history):
        end_date = _raw(row, "endDate")
        revenue = _raw(row, "totalRevenue")
        net_income = _raw(row, "netIncome")
        if end_date and revenue is not None and net_income is not None:
            year = datetime.datetime.utcfromtimestamp(end_date).strftime("%Y")
            years.append({"year": year, "revenue": revenue, "net_income": net_income})
    data["yearly_history"] = years
    if len(years) >= 2 and years[0]["revenue"]:
        data["revenue_cagr"] = (years[-1]["revenue"] / years[0]["revenue"]) ** (
            1.0 / (len(years) - 1)) - 1.0
    else:
        data["revenue_cagr"] = None

    return data


def _score_linear(value, low, high):
    """Maps value linearly onto [0, 100] between the low (->0) and high (->100) benchmarks."""
    if value is None:
        return None
    return float(np.clip((value - low) / (high - low), 0.0, 1.0) * 100.0)


# Benchmark ranges are broad, documented heuristics (not fit to any dataset) used only
# to turn each raw metric into a comparable 0-100 sub-score.
QUALITY_BENCHMARKS = {
    "gross_margin": (0.20, 0.65),
    "operating_margin": (0.05, 0.35),
    "profit_margin": (0.00, 0.25),
    "return_on_equity": (0.05, 0.30),
    "roic": (0.02, 0.20),
    "revenue_growth_yoy": (-0.05, 0.20),
    "earnings_growth_yoy": (-0.10, 0.25),
    "fcf_margin": (0.00, 0.25),
    "current_ratio": (0.75, 2.5),
    "debt_to_equity_inv": (200.0, 0.0),  # inverted: lower D/E -> higher score
}


def compute_quality_score(fund: dict):
    """
    Weighted 0-100 Quality Score across four pillars: profitability, growth,
    cash generation, and balance-sheet health. Each pillar averages the
    sub-scores of the metrics that were actually available (missing metrics
    are excluded, not penalized), so the score degrades gracefully with
    sparser data. Returns (score, pillar_breakdown, weights).
    """
    pillars = {
        "profitability": {
            "Gross margin": _score_linear(fund.get("gross_margin"), *QUALITY_BENCHMARKS["gross_margin"]),
            "Operating margin": _score_linear(fund.get("operating_margin"), *QUALITY_BENCHMARKS["operating_margin"]),
            "Net margin": _score_linear(fund.get("profit_margin"), *QUALITY_BENCHMARKS["profit_margin"]),
            "Return on equity": _score_linear(fund.get("return_on_equity"), *QUALITY_BENCHMARKS["return_on_equity"]),
            "ROIC (approx.)": _score_linear(fund.get("roic"), *QUALITY_BENCHMARKS["roic"]),
        },
        "growth": {
            "Revenue growth YoY": _score_linear(fund.get("revenue_growth_yoy"), *QUALITY_BENCHMARKS["revenue_growth_yoy"]),
            "Earnings growth YoY": _score_linear(fund.get("earnings_growth_yoy"), *QUALITY_BENCHMARKS["earnings_growth_yoy"]),
            "Revenue CAGR (history)": _score_linear(fund.get("revenue_cagr"), *QUALITY_BENCHMARKS["revenue_growth_yoy"]),
        },
        "cash_generation": {
            "FCF margin": _score_linear(fund.get("fcf_margin"), *QUALITY_BENCHMARKS["fcf_margin"]),
            "FCF positive": 100.0 if (fund.get("free_cash_flow") or 0) > 0 else (
                0.0 if fund.get("free_cash_flow") is not None else None),
        },
        "balance_sheet": {
            "Current ratio": _score_linear(fund.get("current_ratio"), *QUALITY_BENCHMARKS["current_ratio"]),
            "Debt / Equity (lower is better)": _score_linear(
                fund.get("debt_to_equity"), *QUALITY_BENCHMARKS["debt_to_equity_inv"]),
        },
    }

    pillar_scores = {}
    for pillar, metrics in pillars.items():
        valid = [v for v in metrics.values() if v is not None]
        pillar_scores[pillar] = float(np.mean(valid)) if valid else None

    weights = {"profitability": 0.35, "growth": 0.25, "cash_generation": 0.20, "balance_sheet": 0.20}
    valid_pillars = {k: v for k, v in pillar_scores.items() if v is not None}
    if not valid_pillars:
        return None, pillars, pillar_scores, weights

    total_weight = sum(weights[k] for k in valid_pillars)
    final_score = sum(weights[k] * v for k, v in valid_pillars.items()) / total_weight
    return float(final_score), pillars, pillar_scores, weights


def fmt_money(value, currency=""):
    if value is None:
        return "n/a"
    abs_v = abs(value)
    sign = "-" if value < 0 else ""
    if abs_v >= 1e12:
        return f"{sign}{abs_v / 1e12:.2f}T {currency}".strip()
    if abs_v >= 1e9:
        return f"{sign}{abs_v / 1e9:.2f}B {currency}".strip()
    if abs_v >= 1e6:
        return f"{sign}{abs_v / 1e6:.2f}M {currency}".strip()
    return f"{sign}{abs_v:,.0f} {currency}".strip()


def fmt_pct(value):
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def fmt_pct0(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def fmt_ratio(value, suffix=""):
    return "n/a" if value is None else f"{value:.2f}{suffix}"
