"""CLI: fetch OHLCV data (Yahoo Finance or Alpha Vantage) and forecast it with Kronos.

Model weights are only downloaded from Hugging Face on the first prediction
(inside KronosForecaster.predict), not when this script starts. Use
--dry-run to validate data fetching and windowing without triggering that
download.

Examples:
    python pipeline/run_forecast.py --ticker AAPL --source yfinance --lookback 400 --pred-len 120
    python pipeline/run_forecast.py --ticker IBM --source alphavantage --interval daily
    python pipeline/run_forecast.py --ticker AAPL --dry-run
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.data_sources import fetch_alphavantage, fetch_yfinance
from pipeline.forecaster import KronosForecaster


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--source", choices=["yfinance", "alphavantage"], default="yfinance")
    parser.add_argument("--interval", default="1d",
                        help="yfinance: 1m/5m/15m/1h/1d/1wk/1mo. alphavantage: 1min/5min/15min/30min/60min/daily/weekly/monthly")
    parser.add_argument("--period", default="2y", help="yfinance lookback period (ignored if --start/--end given)")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--api-key", default=None, help="Alpha Vantage API key (else ALPHAVANTAGE_API_KEY env var)")
    parser.add_argument("--lookback", type=int, default=400, help="Number of historical bars fed to the model")
    parser.add_argument("--pred-len", type=int, default=120, help="Number of future bars to forecast")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--output", default=None, help="CSV path for the forecast (default: pipeline/output/<ticker>_forecast.csv)")
    parser.add_argument("--plot", action="store_true", help="Also save a PNG plot of ground truth vs forecast")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data and validate windowing only; skip the Kronos model download/prediction")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.source == "yfinance":
        df = fetch_yfinance(args.ticker, interval=args.interval, period=args.period, start=args.start, end=args.end)
    else:
        interval = args.interval if args.interval != "1d" else "daily"
        df = fetch_alphavantage(args.ticker, interval=interval, api_key=args.api_key)

    total_needed = args.lookback + args.pred_len
    if len(df) < total_needed:
        raise ValueError(
            f"Only {len(df)} bars available for '{args.ticker}', need at least "
            f"{total_needed} (lookback={args.lookback} + pred_len={args.pred_len}). "
            "Reduce --lookback/--pred-len or widen --period/--start."
        )

    df = df.tail(total_needed).reset_index(drop=True)
    x_df = df.loc[:args.lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[:args.lookback - 1, "timestamps"]
    y_timestamp = df.loc[args.lookback:args.lookback + args.pred_len - 1, "timestamps"]

    if args.dry_run:
        print(f"Dry run OK: fetched {len(df)} bars for '{args.ticker}' from {args.source}.")
        print(x_df.head())
        return

    forecaster = KronosForecaster(tokenizer_name=args.tokenizer, model_name=args.model, max_context=args.max_context)
    pred_df = forecaster.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=args.pred_len,
        T=args.T, top_k=args.top_k, top_p=args.top_p, sample_count=args.sample_count, verbose=True,
    )

    output_path = args.output or f"pipeline/output/{args.ticker}_forecast.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pred_df.to_csv(output_path)
    print(f"Forecast saved to {output_path}")
    print(pred_df.head())

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = os.path.splitext(output_path)[0] + ".png"
        kline_df = df.set_index("timestamps")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(kline_df["close"], label="Ground Truth", color="blue")
        ax1.plot(pred_df["close"], label="Prediction", color="red")
        ax1.set_ylabel("Close")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(kline_df["volume"], label="Ground Truth", color="blue")
        ax2.plot(pred_df["volume"], label="Prediction", color="red")
        ax2.set_ylabel("Volume")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
