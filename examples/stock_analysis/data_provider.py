# -*- coding: utf-8 -*-
"""
data_provider.py

All external data collection lives here: OHLCV price history, fundamentals
(quoteSummary), company profile, and news headlines — all pulled directly
from Yahoo Finance's public endpoints with `requests` (no `yfinance`; its
curl_cffi TLS-impersonation backend does not work through this project's
sandboxed HTTPS proxy). Every other module receives already-parsed Python
data and has no knowledge of the transport.
"""

import time
import datetime
import urllib.parse
import requests
import pandas as pd

YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
CHART_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
QUOTE_SUMMARY_HOSTS = ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]

# Bar interval -> pandas frequency alias, used to build future forecast timestamps.
INTERVAL_TO_FREQ = {
    "1d": "B", "5d": "B", "1wk": "W",
    "1h": "1h", "30m": "30min", "15m": "15min", "5m": "5min", "1m": "1min",
}

DEFAULT_MODULES = (
    "summaryDetail,financialData,defaultKeyStatistics,incomeStatementHistory,"
    "assetProfile,price"
)


class YahooSession:
    """A requests.Session that lazily obtains the cookie + crumb quoteSummary needs."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(YAHOO_HEADERS)
        self._crumb = None

    def crumb(self, force_refresh: bool = False) -> str:
        if self._crumb is None or force_refresh:
            self.session.get("https://fc.yahoo.com", timeout=15)
            resp = self.session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15)
            resp.raise_for_status()
            self._crumb = resp.text.strip()
        return self._crumb


def fetch_price_history(ticker: str, range_: str = "2y", interval: str = "1d",
                         max_retries: int = 3) -> pd.DataFrame:
    """Downloads OHLCV history for `ticker` from Yahoo Finance's chart API."""
    print(f"Fetching {ticker} ({interval}, range={range_}) price history ...")
    params = {"range": range_, "interval": interval, "events": "history"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        result = None
        for host in CHART_HOSTS:
            url = f"https://{host}/v8/finance/chart/{urllib.parse.quote(ticker, safe='')}"
            try:
                resp = requests.get(url, headers=YAHOO_HEADERS, params=params, timeout=20)
                resp.raise_for_status()
                payload = resp.json()["chart"]
                if payload.get("error"):
                    raise ValueError(payload["error"])
                result = payload["result"][0]
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if result is not None:
            break
        print(f"  attempt {attempt}/{max_retries} failed: {last_err}")
        time.sleep(2)

    if result is None:
        raise RuntimeError(f"Failed to fetch price history for {ticker}: {last_err}")

    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s"),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
    })
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0.0)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    meta = result["meta"]
    print(f"  loaded {len(df)} rows, {df['date'].min()} to {df['date'].max()} "
          f"({meta.get('exchangeName', '?')}, {meta.get('currency', '?')})")
    return df


def _raw(node, key, default=None):
    if not isinstance(node, dict):
        return default
    val = node.get(key)
    if isinstance(val, dict):
        return val.get("raw", default)
    return val if val is not None else default


def fetch_quote_summary(ticker: str, modules: str = DEFAULT_MODULES,
                         session: "YahooSession" = None, max_retries: int = 3) -> dict:
    """
    Fetches the requested quoteSummary modules for `ticker` and returns the raw
    (module_name -> module_dict) mapping. Values inside each module are still in
    Yahoo's {"raw": x, "fmt": "..."} wrapper; use `_raw()` or module-specific
    parsers to unwrap them. Returns {} on failure so callers can degrade gracefully.
    """
    session = session or YahooSession()
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            crumb = session.crumb(force_refresh=(attempt > 1))
            for host in QUOTE_SUMMARY_HOSTS:
                url = f"https://{host}/v10/finance/quoteSummary/{urllib.parse.quote(ticker, safe='')}"
                resp = session.session.get(url, params={"modules": modules, "crumb": crumb},
                                            timeout=20)
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
                    return {}
                return results[0]
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2)
    print(f"  warning: failed to fetch fundamentals for {ticker}: {last_err}")
    return {}


def fetch_news(ticker: str, count: int = 8, max_retries: int = 3) -> list:
    """Returns recent news items for `ticker`: [{title, publisher, link, published, summary}, ...]."""
    params = {"q": ticker, "newsCount": count, "quotesCount": 0}
    last_err = None
    for attempt in range(1, max_retries + 1):
        for host in CHART_HOSTS:
            try:
                resp = requests.get(f"https://{host}/v1/finance/search", headers=YAHOO_HEADERS,
                                     params=params, timeout=20)
                resp.raise_for_status()
                items = resp.json().get("news", [])
                news = []
                for item in items[:count]:
                    ts = item.get("providerPublishTime")
                    news.append({
                        "title": item.get("title", ""),
                        "publisher": item.get("publisher", ""),
                        "link": item.get("link", ""),
                        "published": datetime.datetime.utcfromtimestamp(ts) if ts else None,
                        "summary": item.get("summary", "") or item.get("title", ""),
                    })
                return news
            except Exception as e:  # noqa: BLE001
                last_err = e
        time.sleep(2)
    print(f"  warning: failed to fetch news for {ticker}: {last_err}")
    return []
