# -*- coding: utf-8 -*-
"""
prediction_yahoo_finance.py

Description:
    Forecasts future price/volume data for any Yahoo Finance ticker (e.g. AAPL, MSFT,
    NVDA, SAP.DE) using the Kronos model. Historical OHLCV data is downloaded directly
    from Yahoo Finance's public chart API and fed into Kronos to autoregressively
    predict the next `pred_len` bars.

Usage:
    python prediction_yahoo_finance.py --ticker AAPL
    python prediction_yahoo_finance.py --ticker SAP.DE --lookback 300 --pred_len 60
    python prediction_yahoo_finance.py --ticker NVDA --interval 1h --range 60d

Output:
    - Saves the prediction results to ./outputs/pred_<ticker>_data.csv
    - Saves a chart to ./outputs/pred_<ticker>_chart.png
"""

import os
import time
import argparse
import sys

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

YAHOO_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Interval strings that pandas can use to build a matching future date range.
INTERVAL_TO_FREQ = {
    "1d": "B", "5d": "B", "1wk": "W",
    "1h": "1h", "30m": "30min", "15m": "15min", "5m": "5min", "1m": "1min",
}


def fetch_yahoo_data(ticker: str, range_: str, interval: str, max_retries: int = 3) -> pd.DataFrame:
    """Downloads OHLCV history for `ticker` from Yahoo Finance's chart API."""
    print(f"Fetching {ticker} ({interval}, range={range_}) from Yahoo Finance ...")
    params = {"range": range_, "interval": interval, "events": "history"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        for host in YAHOO_HOSTS:
            url = f"https://{host}/v8/finance/chart/{ticker}"
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
                result = None
        if result is not None:
            break
        print(f"  attempt {attempt}/{max_retries} failed: {last_err}")
        time.sleep(2)

    if result is None:
        print(f"Failed to fetch data for {ticker}: {last_err}")
        sys.exit(1)

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

    currency = result["meta"].get("currency", "?")
    exchange = result["meta"].get("exchangeName", "?")
    print(f"Data loaded: {len(df)} rows, {df['date'].min()} to {df['date'].max()} "
          f"({exchange}, {currency})")
    return df


def prepare_inputs(df: pd.DataFrame, lookback: int, pred_len: int, interval: str):
    if len(df) < lookback:
        raise ValueError(
            f"Not enough history: got {len(df)} rows, need at least {lookback}. "
            f"Try a larger --range."
        )
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.iloc[-lookback:]["date"]

    freq = INTERVAL_TO_FREQ.get(interval, "B")
    last_ts = df["date"].iloc[-1]
    y_timestamp = pd.date_range(start=last_ts, periods=pred_len + 1, freq=freq)[1:]
    return x_df, pd.Series(x_timestamp).reset_index(drop=True), pd.Series(y_timestamp)


def plot_result(df_hist: pd.DataFrame, df_pred: pd.DataFrame, ticker: str, lookback: int):
    hist_tail = df_hist.iloc[-lookback:]
    plt.figure(figsize=(12, 6))
    plt.plot(hist_tail["date"], hist_tail["close"], label="Historical", color="blue")
    plt.plot(df_pred["date"], df_pred["close"], label="Predicted", color="red", linestyle="--")
    plt.title(f"Kronos Forecast for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"pred_{ticker.replace('.', '_')}_chart.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Chart saved: {plot_path}")
    return plot_path


def predict_future(ticker, lookback, pred_len, range_, interval, tokenizer_name, model_name,
                    device, T, top_p, sample_count):
    print(f"Loading Kronos tokenizer:{tokenizer_name} model:{model_name} ...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max(512, lookback))

    df = fetch_yahoo_data(ticker, range_, interval)
    x_df, x_timestamp, y_timestamp = prepare_inputs(df, lookback, pred_len, interval)

    print("Generating predictions ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=True,
    )
    pred_df["date"] = y_timestamp.values

    df_out = pd.concat([
        df[["date", "open", "high", "low", "close", "volume", "amount"]],
        pred_df[["date", "open", "high", "low", "close", "volume", "amount"]],
    ]).reset_index(drop=True)

    out_file = os.path.join(SAVE_DIR, f"pred_{ticker.replace('.', '_')}_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Prediction completed and saved: {out_file}")

    print("Forecasted Data Head:")
    print(pred_df.head())

    plot_result(df, pred_df, ticker, lookback)
    return pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos forecast for any Yahoo Finance ticker")
    parser.add_argument("--ticker", type=str, required=True,
                         help="Yahoo Finance ticker symbol, e.g. AAPL, MSFT, NVDA, SAP.DE")
    parser.add_argument("--lookback", type=int, default=400, help="Number of historical bars fed to the model")
    parser.add_argument("--pred_len", type=int, default=120, help="Number of future bars to predict")
    parser.add_argument("--range", type=str, default="2y", dest="range_",
                         help="Yahoo Finance history range to download (e.g. 1y, 2y, 60d)")
    parser.add_argument("--interval", type=str, default="1d",
                         help="Bar interval: 1d, 1h, 30m, 15m, 5m, 1m, 1wk, ...")
    parser.add_argument("--tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--model", type=str, default="NeoQuasar/Kronos-small",
                         help="NeoQuasar/Kronos-mini | -small | -base | -large")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda:0, mps (auto-detected if omitted)")
    parser.add_argument("--T", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--sample_count", type=int, default=1, help="Number of samples averaged per prediction")
    args = parser.parse_args()

    predict_future(
        ticker=args.ticker,
        lookback=args.lookback,
        pred_len=args.pred_len,
        range_=args.range_,
        interval=args.interval,
        tokenizer_name=args.tokenizer,
        model_name=args.model,
        device=args.device,
        T=args.T,
        top_p=args.top_p,
        sample_count=args.sample_count,
    )
