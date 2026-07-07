# -*- coding: utf-8 -*-
"""
fundamentals.py

Fetches fundamental data (valuation, growth, cash flow, leverage, margins) for a
Yahoo Finance ticker via the quoteSummary API. This endpoint requires a session
cookie + "crumb" token (Yahoo's CSRF guard), obtained once per run and reused.
"""

import time
import requests

YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
QUOTE_SUMMARY_HOSTS = ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]
MODULES = "summaryDetail,financialData,defaultKeyStatistics,incomeStatementHistory,price"


def _get_crumb(session: requests.Session) -> str:
    session.get("https://fc.yahoo.com", timeout=15)  # sets the initial session cookie
    resp = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15)
    resp.raise_for_status()
    return resp.text.strip()


def _raw(node, key, default=None):
    if not isinstance(node, dict):
        return default
    val = node.get(key)
    if isinstance(val, dict):
        return val.get("raw", default)
    return val if val is not None else default


def fetch_fundamentals(ticker: str, max_retries: int = 3) -> dict:
    """
    Returns a flat dict of fundamental metrics for `ticker`, or an empty dict
    (with an "error" key) if Yahoo has no fundamentals for this symbol (e.g. some
    indices/crypto) or the request ultimately fails.
    """
    session = requests.Session()
    session.headers.update(YAHOO_HEADERS)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            crumb = _get_crumb(session)
            for host in QUOTE_SUMMARY_HOSTS:
                url = f"https://{host}/v10/finance/quoteSummary/{ticker}"
                resp = session.get(url, params={"modules": MODULES, "crumb": crumb}, timeout=20)
                if resp.status_code == 401:
                    last_err = "Unauthorized (stale crumb)"
                    continue
                resp.raise_for_status()
                payload = resp.json()["quoteSummary"]
                if payload.get("error"):
                    last_err = payload["error"]
                    continue
                results = payload.get("result")
                if not results:
                    return {"error": "No fundamentals data available for this ticker"}
                return _parse(results[0])
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2)

    return {"error": f"Failed to fetch fundamentals: {last_err}"}


def _parse(result: dict) -> dict:
    fin = result.get("financialData", {})
    stats = result.get("defaultKeyStatistics", {})
    summ = result.get("summaryDetail", {})
    price = result.get("price", {})

    data = {
        "currency": _raw(price, "currency"),
        "market_cap": _raw(price, "marketCap"),
        "trailing_pe": _raw(summ, "trailingPE"),
        "forward_pe": _raw(stats, "forwardPE") or _raw(summ, "forwardPE"),
        "peg_ratio": _raw(stats, "pegRatio"),
        "price_to_book": _raw(stats, "priceToBook"),
        "ev_to_ebitda": _raw(stats, "enterpriseToEbitda"),
        "ev_to_revenue": _raw(stats, "enterpriseToRevenue"),
        "trailing_eps": _raw(stats, "trailingEps"),
        "forward_eps": _raw(stats, "forwardEps"),
        "dividend_yield": _raw(summ, "dividendYield"),
        "beta": _raw(summ, "beta") or _raw(stats, "beta"),

        "revenue_growth_yoy": _raw(fin, "revenueGrowth"),
        "earnings_growth_yoy": _raw(fin, "earningsGrowth"),
        "revenue_quarterly_growth": _raw(stats, "revenueQuarterlyGrowth"),
        "earnings_quarterly_growth": _raw(stats, "earningsQuarterlyGrowth"),

        "total_revenue": _raw(fin, "totalRevenue"),
        "gross_margin": _raw(fin, "grossMargins"),
        "operating_margin": _raw(fin, "operatingMargins"),
        "profit_margin": _raw(fin, "profitMargins"),
        "return_on_equity": _raw(fin, "returnOnEquity"),
        "return_on_assets": _raw(fin, "returnOnAssets"),

        "free_cash_flow": _raw(fin, "freeCashflow"),
        "operating_cash_flow": _raw(fin, "operatingCashflow"),
        "total_cash": _raw(fin, "totalCash"),
        "total_debt": _raw(fin, "totalDebt"),
        "debt_to_equity": _raw(fin, "debtToEquity"),
        "current_ratio": _raw(fin, "currentRatio"),
        "quick_ratio": _raw(fin, "quickRatio"),

        "target_mean_price": _raw(fin, "targetMeanPrice"),
        "recommendation": _raw(fin, "recommendationKey"),
        "num_analyst_opinions": _raw(fin, "numberOfAnalystOpinions"),
    }

    if data["total_debt"] is not None and data["total_cash"] is not None:
        data["net_debt"] = data["total_debt"] - data["total_cash"]
    else:
        data["net_debt"] = None

    # Multi-year revenue/net-income trend (oldest -> newest); Yahoo's other legacy
    # statement fields (balance sheet, historical cash flow) are no longer populated.
    history = result.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])
    years = []
    for row in reversed(history):
        end_date = _raw(row, "endDate")
        revenue = _raw(row, "totalRevenue")
        net_income = _raw(row, "netIncome")
        if end_date and revenue is not None and net_income is not None:
            years.append({"year": pd_to_year(end_date), "revenue": revenue, "net_income": net_income})
    data["yearly_history"] = years
    return data


def pd_to_year(epoch_seconds: int) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(epoch_seconds).strftime("%Y")


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
