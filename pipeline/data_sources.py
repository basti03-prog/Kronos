"""Fetch OHLCV market data and normalize it to the schema
`KronosPredictor.predict()` expects: a DataFrame with
`['open', 'high', 'low', 'close']` (plus optional `volume`/`amount`) and a
matching `timestamps` column of the same length.

Network access:
  - fetch_yfinance() talks to Yahoo Finance (query1/query2.finance.yahoo.com).
    No API key required.
  - fetch_alphavantage() talks to www.alphavantage.co and requires a free
    API key (https://www.alphavantage.co/support/#api-key), passed via
    api_key= or the ALPHAVANTAGE_API_KEY environment variable.
"""
import os
from typing import Optional

import pandas as pd
import requests

REQUIRED_COLUMNS = ["timestamps", "open", "high", "low", "close"]


def fetch_yfinance(ticker: str, interval: str = "1d", period: str = "2y",
                    start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf

    ticker_obj = yf.Ticker(ticker)
    if start or end:
        raw = ticker_obj.history(interval=interval, start=start, end=end, auto_adjust=False)
    else:
        raw = ticker_obj.history(interval=interval, period=period, auto_adjust=False)

    if raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for ticker '{ticker}'.")

    df = raw.reset_index()
    timestamp_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        timestamp_col: "timestamps",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    return df[REQUIRED_COLUMNS + ["volume", "amount"]]


_ALPHAVANTAGE_FUNCTIONS = {
    "daily": "TIME_SERIES_DAILY",
    "weekly": "TIME_SERIES_WEEKLY",
    "monthly": "TIME_SERIES_MONTHLY",
    "1min": "TIME_SERIES_INTRADAY",
    "5min": "TIME_SERIES_INTRADAY",
    "15min": "TIME_SERIES_INTRADAY",
    "30min": "TIME_SERIES_INTRADAY",
    "60min": "TIME_SERIES_INTRADAY",
}


def fetch_alphavantage(ticker: str, interval: str = "daily",
                        api_key: Optional[str] = None, outputsize: str = "full") -> pd.DataFrame:
    api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Alpha Vantage requires an API key. Get a free one at "
            "https://www.alphavantage.co/support/#api-key and either pass "
            "api_key=... or set the ALPHAVANTAGE_API_KEY environment variable."
        )

    if interval not in _ALPHAVANTAGE_FUNCTIONS:
        raise ValueError(f"Unsupported interval '{interval}'. Choose one of {sorted(_ALPHAVANTAGE_FUNCTIONS)}.")

    function = _ALPHAVANTAGE_FUNCTIONS[interval]
    params = {
        "function": function,
        "symbol": ticker,
        "outputsize": outputsize,
        "apikey": api_key,
        "datatype": "json",
    }
    if function == "TIME_SERIES_INTRADAY":
        params["interval"] = interval

    response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    series_key = next(
        (k for k in payload if k.startswith(("Time Series", "Weekly", "Monthly"))), None
    )
    if series_key is None:
        raise ValueError(f"Alpha Vantage returned no time series for '{ticker}': {payload}")

    raw = pd.DataFrame(payload[series_key]).T
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    raw = raw.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume",
    })
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = raw[col].astype(float)

    df = raw.reset_index().rename(columns={"index": "timestamps"})
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    return df[REQUIRED_COLUMNS + ["volume", "amount"]]
