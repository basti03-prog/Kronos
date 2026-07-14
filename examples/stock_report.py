# -*- coding: utf-8 -*-
"""
stock_report.py

Complete stock analysis tool built on Kronos: price forecasting, technical
analysis (RSI/MACD/EMA/Bollinger/ATR/ADX/support-resistance/Golden-Death
Cross), fundamental analysis (growth/margins/ROIC/leverage + a 0-100 Quality
Score), valuation (DCF + Graham fair value, margin of safety), risk analysis
(volatility/beta/max drawdown/Sharpe/Sortino), news & sentiment, a data-driven
company-quality assessment (moat/market position/growth/risks), and a final
AI investment decision layer: a weighted Investment Score (0-100), a 5-tier
STRONG BUY..SELL rating, Buy/Hold/Sell probabilities, a confidence score, and
the strongest bullish/bearish factors behind the call.

All the actual logic lives in the stock_analysis/ package (one module per
concern — data_provider, technical, fundamental_analysis, valuation, risk,
news_sentiment, quality, prediction, scoring, ai_decision, report); this
script only orchestrates the pipeline and exposes the CLI.

Usage:
    python stock_report.py --ticker AAPL
    python stock_report.py --ticker SAP.DE --pred_len 40 --ensemble_size 8

Output (./outputs/):
    - report_<TICKER>.pdf              6-page professional PDF report
    - report_<TICKER>_overview.png     quick-look PNG of the overview page
    - report_<TICKER>_interactive.html self-contained interactive Plotly chart

The AI investment decision is also printed to the terminal at the end of
every run.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stock_analysis import data_provider as dp
from stock_analysis import technical as ta
from stock_analysis import fundamental_analysis as fa
from stock_analysis import valuation as val
from stock_analysis import risk as rk
from stock_analysis import news_sentiment as ns
from stock_analysis import quality as qual
from stock_analysis import prediction as pred
from stock_analysis import scoring as sc
from stock_analysis import ai_decision as ai
from stock_analysis import report as rpt

SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def run_analysis(ticker, lookback, pred_len, range_, interval, tokenizer_name, model_name,
                  device, T, top_p, ensemble_size, sr_window, sr_tolerance, risk_free_rate,
                  benchmark_ticker, news_count, skip_fundamentals, skip_news):
    predictor = pred.load_predictor(tokenizer_name, model_name, device, max(512, lookback))

    df = dp.fetch_price_history(ticker, range_, interval)
    if len(df) < lookback:
        raise ValueError(f"Not enough history: got {len(df)} rows, need at least {lookback}. "
                          f"Try a larger --range.")

    print("Computing technical indicators ...")
    tech = ta.TechnicalSnapshot(df, lookback, interval, sr_window=sr_window, sr_tolerance=sr_tolerance)

    print("Fetching benchmark index for beta ...")
    try:
        bench_df = dp.fetch_price_history(benchmark_ticker, range_, interval)
    except Exception as e:  # noqa: BLE001
        print(f"  warning: benchmark fetch failed ({e}); beta will be n/a")
        bench_df = None
    risk_snap = rk.RiskSnapshot(df, lookback, interval, bench_df, risk_free_rate)

    if skip_fundamentals:
        fund = {"error": "Fundamentals skipped (--skip_fundamentals)"}
    else:
        print("Fetching fundamentals ...")
        session = dp.YahooSession()
        qs_raw = dp.fetch_quote_summary(ticker, session=session)
        fund = fa.parse_fundamentals(qs_raw)
        if fund.get("error"):
            print(f"  warning: {fund['error']}")

    quality_score, quality_pillars, quality_pillar_scores, quality_weights = (
        fa.compute_quality_score(fund) if not fund.get("error") else (None, {}, {}, {})
    )

    last_close = float(df["close"].iloc[-1])
    dcf = None
    graham = None
    mos_dcf = mos_graham = blended_fv = None
    if not fund.get("error"):
        dcf = val.estimate_dcf_fair_value(
            fund.get("free_cash_flow"), fund.get("shares_outstanding"), fund.get("net_debt"),
            fund.get("revenue_growth_yoy"), beta=risk_snap.beta or fund.get("beta") or 1.0)
        graham = val.graham_number(fund.get("trailing_eps"), fund.get("book_value_per_share"))
        mos_dcf = val.margin_of_safety(last_close, dcf["fair_value"] if dcf else None)
        mos_graham = val.margin_of_safety(last_close, graham)
        blended_fv = val.blended_fair_value(
            [dcf["fair_value"] if dcf else None, graham])

    if skip_news:
        news = {"items": [], "avg_compound": 0.0, "overall_label": "neutral",
                "counts": {"positive": 0, "neutral": 0, "negative": 0}}
    else:
        print("Fetching news & scoring sentiment ...")
        news_items = dp.fetch_news(ticker, count=news_count)
        news = ns.analyze_news(news_items)

    quality_assessment = qual.assess_company_quality(
        fund, risk_snap, {"margin_of_safety_dcf": mos_dcf, "margin_of_safety_graham": mos_graham})

    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.iloc[-lookback:]["date"].reset_index(drop=True)
    freq = dp.INTERVAL_TO_FREQ.get(interval, "B")
    y_timestamp = pd.date_range(start=df["date"].iloc[-1], periods=pred_len + 1, freq=freq)[1:]

    print(f"Running {ensemble_size}-member Monte-Carlo ensemble forecast ({pred_len} steps each) ...")
    close_paths = pred.run_ensemble_forecast(predictor, x_df, x_timestamp, y_timestamp, pred_len,
                                              T, top_p, ensemble_size)
    p_ensemble_up = float(np.mean(close_paths[:, -1] > last_close))
    forecast_expected_return = float(np.mean(close_paths[:, -1])) / last_close - 1.0

    pred_score, pred_label = sc.prediction_score(p_ensemble_up)
    tech_score, tech_label, tech_components, tech_weights = sc.technical_score(
        tech.ema50.iloc[-1], tech.ema200.iloc[-1], last_close, tech.hist.iloc[-1], tech.rsi.iloc[-1])

    investment_score, recommendation, score_breakdown, score_weights, reasons = sc.compute_investment_score(
        pred_score, quality_score, mos_dcf, tech_score, risk_snap.sharpe, news["avg_compound"])

    print(f"Prediction: {pred_label} ({pred_score:.0f}/100)  |  Technicals: {tech_label} "
          f"({tech_score:.0f}/100)  |  Investment score: "
          + (f"{investment_score:.0f}/100 -> {recommendation}" if investment_score is not None
             else f"n/a -> {recommendation}"))

    ctx = {
        "ticker": ticker, "df": df, "lookback": lookback, "interval": interval,
        "pred_len": pred_len, "y_timestamp": y_timestamp, "close_paths": close_paths,
        "last_close": last_close, "tech": tech, "fund": fund,
        "quality_score": quality_score, "quality_pillars": quality_pillars,
        "quality_pillar_scores": quality_pillar_scores, "quality_weights": quality_weights,
        "dcf": dcf, "graham": graham, "margin_of_safety_dcf": mos_dcf,
        "margin_of_safety_graham": mos_graham, "blended_fair_value": blended_fv,
        "target_mean_price": fund.get("target_mean_price"),
        "risk": risk_snap, "news": news, "quality_assessment": quality_assessment,
        "p_ensemble_up": p_ensemble_up, "forecast_expected_return": forecast_expected_return,
        "pred_score": pred_score, "pred_label": pred_label,
        "tech_score": tech_score, "tech_label": tech_label,
        "tech_components": tech_components, "tech_weights": tech_weights,
        "investment_score": investment_score, "recommendation_3tier": recommendation,
        "score_breakdown": score_breakdown, "score_weights": score_weights, "reasons": reasons,
    }

    print("Computing AI investment decision ...")
    decision = ai.build_decision(ctx)
    ctx["decision"] = decision
    ctx["recommendation"] = decision["rating"]  # 5-tier STRONG BUY..SELL drives the report banners

    print()
    print(ai.format_terminal_summary(ticker, decision))
    print()

    safe_ticker = ticker.replace(".", "_").replace("^", "")
    pdf_path = os.path.join(SAVE_DIR, f"report_{safe_ticker}.pdf")
    png_path = os.path.join(SAVE_DIR, f"report_{safe_ticker}_overview.png")
    html_path = os.path.join(SAVE_DIR, f"report_{safe_ticker}_interactive.html")

    print("Rendering PDF + overview PNG ...")
    rpt.build_pdf_and_png(ctx, pdf_path, png_path)
    print("Rendering interactive HTML chart ...")
    rpt.build_interactive_html(ctx, html_path)

    print(f"Report saved: {pdf_path}")
    print(f"Report saved: {png_path}")
    print(f"Report saved: {html_path}")
    return ctx, pdf_path, png_path, html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Kronos-based stock analysis tool")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred_len", type=int, default=60)
    parser.add_argument("--range", type=str, default="2y", dest="range_")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--model", type=str, default="NeoQuasar/Kronos-small")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--sr_window", type=int, default=5)
    parser.add_argument("--sr_tolerance", type=float, default=0.015)
    parser.add_argument("--risk_free_rate", type=float, default=0.04,
                         help="Annual risk-free rate used for Sharpe/Sortino (default 4%%)")
    parser.add_argument("--benchmark", type=str, default="^GSPC", dest="benchmark_ticker",
                         help="Benchmark index ticker used to compute beta (default S&P 500)")
    parser.add_argument("--news_count", type=int, default=8)
    parser.add_argument("--skip_fundamentals", action="store_true")
    parser.add_argument("--skip_news", action="store_true")
    args = parser.parse_args()

    run_analysis(
        ticker=args.ticker, lookback=args.lookback, pred_len=args.pred_len, range_=args.range_,
        interval=args.interval, tokenizer_name=args.tokenizer, model_name=args.model,
        device=args.device, T=args.T, top_p=args.top_p, ensemble_size=args.ensemble_size,
        sr_window=args.sr_window, sr_tolerance=args.sr_tolerance,
        risk_free_rate=args.risk_free_rate, benchmark_ticker=args.benchmark_ticker,
        news_count=args.news_count, skip_fundamentals=args.skip_fundamentals,
        skip_news=args.skip_news,
    )
