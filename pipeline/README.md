# Kronos Data Pipeline

Fetches real stock data (Yahoo Finance or Alpha Vantage) and forecasts it with
the official Kronos model, using the unmodified `KronosPredictor` API from
`model/kronos.py`.

## Install

```shell
pip install -r requirements.txt              # official Kronos deps
pip install -r requirements-pipeline.txt      # extra: yfinance
```

## Lazy model loading

`KronosForecaster` (in `forecaster.py`) does not download anything at
construction time. `Kronos.from_pretrained(...)` /
`KronosTokenizer.from_pretrained(...)` are only called on the first
`.predict()` call, so building a pipeline or running `--dry-run` never hits
the Hugging Face Hub.

## Usage

```shell
# Yahoo Finance, no API key needed
python pipeline/run_forecast.py --ticker AAPL --source yfinance --lookback 400 --pred-len 120 --plot

# Alpha Vantage, requires a free API key
export ALPHAVANTAGE_API_KEY=your_key_here
python pipeline/run_forecast.py --ticker IBM --source alphavantage --interval daily

# Validate data fetching/windowing without downloading the model
python pipeline/run_forecast.py --ticker AAPL --dry-run
```

Results are written to `pipeline/output/<ticker>_forecast.csv` (and
`<ticker>_forecast.png` with `--plot`).

## Network requirements

| Target                                    | Needed for                    | Auth              |
|--------------------------------------------|--------------------------------|--------------------|
| `huggingface.co` (+ `*.huggingface.co`)    | downloading Kronos model weights | none (public repos) |
| `query1.finance.yahoo.com`, `query2.finance.yahoo.com` | `--source yfinance`        | none               |
| `www.alphavantage.co`                     | `--source alphavantage`        | `ALPHAVANTAGE_API_KEY` |

In sandboxed environments (e.g. Claude Code on the web) these hosts may need
to be added to the environment's network allowlist before a real run
succeeds; `--dry-run` still works without any of them except the data
source's own host.
