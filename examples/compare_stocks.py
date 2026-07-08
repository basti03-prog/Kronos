# -*- coding: utf-8 -*-
"""
compare_stocks.py

Runs the full stock_report.py pipeline for a list of tickers and renders a
single side-by-side comparison dashboard: indexed price performance,
Investment Score + component breakdown, Buy/Hold/Sell probabilities, and a
key-metrics table.

Usage:
    python compare_stocks.py --tickers MCD NFLX ADBE
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stock_report import run_analysis
from stock_analysis import palette as pal

SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

SERIES_COLORS = [pal.BLUE, pal.ORANGE, pal.VIOLET, pal.AQUA, pal.MAGENTA]


def fmt_money(x):
    if x is None:
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9:
        return f"{sign}{x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{sign}{x / 1e6:.2f}M"
    return f"{sign}{x:,.0f}"


def compare(tickers, **kwargs):
    results = {}
    for i, ticker in enumerate(tickers):
        print(f"\n=== [{i + 1}/{len(tickers)}] {ticker} ===")
        ctx, *_ = run_analysis(ticker=ticker, **kwargs)
        results[ticker] = ctx

    colors = {t: SERIES_COLORS[i % len(SERIES_COLORS)] for i, t in enumerate(tickers)}

    fig = plt.figure(figsize=(13, 15), facecolor=pal.SURFACE)
    gs = GridSpec(4, 2, height_ratios=[1.7, 1.6, 1.3, 2.0], hspace=0.6, wspace=0.3,
                  left=0.08, right=0.96, top=0.88, bottom=0.04, figure=fig)

    fig.suptitle(" vs ".join(tickers) + " — Side-by-Side Comparison", fontsize=19,
                 fontweight="bold", color=pal.INK, x=0.08, ha="left", y=0.975)
    fig.text(0.08, 0.945, "Kronos Complete Stock Analysis Report — same pipeline, same weights, "
                          "run independently per ticker", fontsize=9.3, color=pal.MUTED)

    # --- Indexed price performance ---
    ax_price = fig.add_subplot(gs[0, :])
    for t in tickers:
        hist = results[t]["df"].iloc[-results[t]["lookback"]:].reset_index(drop=True)
        indexed = hist["close"] / hist["close"].iloc[0] * 100.0
        ax_price.plot(hist["date"], indexed, color=colors[t], linewidth=2, label=t)
    ax_price.axhline(100, color=pal.BASELINE, linewidth=1, linestyle=":")
    ax_price.set_ylabel("Indexed price (start = 100)", color=pal.INK_SECONDARY, fontsize=10)
    ax_price.legend(loc="upper left", fontsize=10, frameon=False, ncol=len(tickers))
    ax_price.set_title("Historical performance (indexed)", fontsize=11.5, fontweight="bold",
                        color=pal.INK, loc="left")
    pal.style_axes(ax_price)
    for spine in ("top", "right"):
        ax_price.spines[spine].set_visible(False)

    # --- Investment score + components ---
    ax_score = fig.add_subplot(gs[1, 0])
    scores = [results[t]["investment_score"] or 0 for t in tickers]
    bar_colors = [pal.VERDICT_COLOR.get(results[t]["decision"]["rating"], pal.WARNING) for t in tickers]
    y = np.arange(len(tickers))
    ax_score.barh(y, scores, color=bar_colors, height=0.5)
    for yi, t, s in zip(y, tickers, scores):
        rating = results[t]["decision"]["rating"]
        ax_score.text(s + 2, yi, f"{s:.0f}  ({rating})", va="center", fontsize=9.5,
                       color=pal.INK_SECONDARY)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(tickers, fontsize=11, color=pal.INK_SECONDARY, fontweight="bold")
    ax_score.invert_yaxis()
    ax_score.set_xlim(0, 115)
    ax_score.axvline(50, color=pal.BASELINE, linewidth=1, linestyle=":")
    ax_score.set_title("Investment Score", fontsize=11.5, fontweight="bold", color=pal.INK, loc="left")
    pal.style_axes(ax_score)
    ax_score.spines["left"].set_visible(False)
    ax_score.tick_params(left=False)
    for spine in ("top", "right"):
        ax_score.spines[spine].set_visible(False)

    ax_comp = fig.add_subplot(gs[1, 1])
    categories = list(results[tickers[0]]["score_breakdown"].keys())
    n_cat = len(categories)
    bar_h = 0.8 / len(tickers)
    for i, t in enumerate(tickers):
        vals = [results[t]["score_breakdown"].get(c) or 0 for c in categories]
        y_pos = np.arange(n_cat) + i * bar_h
        ax_comp.barh(y_pos, vals, height=bar_h * 0.9, color=colors[t], label=t)
        for yi, v in zip(y_pos, vals):
            ax_comp.text(max(v, 2) + 2, yi, f"{v:.0f}", va="center", fontsize=6.8,
                          color=pal.INK_SECONDARY)
    ax_comp.set_yticks(np.arange(n_cat) + bar_h * (len(tickers) - 1) / 2)
    ax_comp.set_yticklabels([c.replace(" (Kronos)", "") for c in categories], fontsize=8.5,
                             color=pal.INK_SECONDARY)
    ax_comp.invert_yaxis()
    ax_comp.set_xlim(0, 112)
    ax_comp.axvline(50, color=pal.BASELINE, linewidth=1, linestyle=":")
    ax_comp.legend(loc="lower right", fontsize=8, frameon=False, ncol=len(tickers))
    ax_comp.set_title("Score components", fontsize=11.5, fontweight="bold", color=pal.INK, loc="left")
    pal.style_axes(ax_comp)
    ax_comp.spines["left"].set_visible(False)
    ax_comp.tick_params(left=False)
    for spine in ("top", "right"):
        ax_comp.spines[spine].set_visible(False)

    # --- Buy/Hold/Sell probabilities ---
    ax_bhs = fig.add_subplot(gs[2, :])
    width = 0.8 / len(tickers)
    x = np.arange(3)
    for i, t in enumerate(tickers):
        p = results[t]["decision"]["probabilities"]
        vals = [p["buy"], p["hold"], p["sell"]]
        ax_bhs.bar(x + i * width, vals, width=width * 0.9, color=colors[t], label=t)
    ax_bhs.set_xticks(x + width * (len(tickers) - 1) / 2)
    ax_bhs.set_xticklabels(["Buy", "Hold", "Sell"], fontsize=10.5, color=pal.INK_SECONDARY)
    ax_bhs.set_ylabel("Probability (%)", color=pal.INK_SECONDARY, fontsize=9.5)
    ax_bhs.legend(loc="upper right", fontsize=9, frameon=False, ncol=len(tickers))
    ax_bhs.set_title("Buy / Hold / Sell probability", fontsize=11.5, fontweight="bold",
                      color=pal.INK, loc="left")
    pal.style_axes(ax_bhs)
    for spine in ("top", "right"):
        ax_bhs.spines[spine].set_visible(False)

    # --- Key metrics table ---
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")
    ax_table.text(0.0, 1.0, "Key metrics", transform=ax_table.transAxes, fontsize=12,
                   fontweight="bold", color=pal.INK, va="top")

    rows = [
        ("Last close", lambda r: f"${r['last_close']:,.2f}"),
        ("Investment Score", lambda r: f"{r['investment_score']:.0f}/100 -> {r['decision']['rating']}"),
        ("Confidence", lambda r: f"{r['decision']['confidence']:.0f}%"),
        ("Kronos P[price up]", lambda r: f"{r['p_ensemble_up'] * 100:.0f}%"),
        ("Avg. projected move", lambda r: f"{r['forecast_expected_return'] * 100:+.1f}%"),
        ("DCF margin of safety", lambda r: (f"{r['margin_of_safety_dcf'] * 100:+.0f}%"
                                             if r['margin_of_safety_dcf'] is not None else "n/a")),
        ("FCF margin", lambda r: (f"{r['fund'].get('free_cash_flow') / r['fund'].get('total_revenue') * 100:.0f}%"
                                   if r['fund'].get('free_cash_flow') and r['fund'].get('total_revenue') else "n/a")),
        ("Operating / Net margin", lambda r: (
            f"{(r['fund'].get('operating_margin') or 0) * 100:.0f}% / "
            f"{(r['fund'].get('profit_margin') or 0) * 100:.0f}%")),
        ("PEG ratio", lambda r: (f"{r['fund'].get('peg_ratio'):.2f}" if r['fund'].get('peg_ratio') else "n/a")),
        ("Beta (vs S&P 500)", lambda r: (f"{r['risk'].beta:.2f}" if r['risk'].beta is not None else "n/a")),
        ("Sharpe ratio", lambda r: f"{r['risk'].sharpe:.2f}" if r['risk'].sharpe is not None else "n/a"),
        ("Max drawdown", lambda r: f"{r['risk'].max_drawdown_pct:.0f}%"),
        ("News sentiment", lambda r: r['news']['overall_label']),
    ]

    n_rows = len(rows)
    col_w = 1.0 / (len(tickers) + 1)
    row_h = 0.85 / n_rows
    y0 = 0.90
    for label, _ in rows:
        ax_table.text(0.0, y0, label, transform=ax_table.transAxes, fontsize=9,
                       color=pal.MUTED, va="top")
        y0 -= row_h
    for i, t in enumerate(tickers):
        y0 = 0.90
        ax_table.text(col_w * (i + 1) + 0.02, 0.98, t, transform=ax_table.transAxes, fontsize=10.5,
                       fontweight="bold", color=colors[t], va="top")
        for label, fn in rows:
            try:
                val = fn(results[t])
            except Exception:  # noqa: BLE001
                val = "n/a"
            ax_table.text(col_w * (i + 1) + 0.02, y0, str(val), transform=ax_table.transAxes,
                           fontsize=9, color=pal.INK_SECONDARY, va="top")
            y0 -= row_h

    fig.text(0.08, 0.012,
              "Disclaimer: generated by a machine-learning model (Kronos) and rule-based financial "
              "heuristics for research/educational purposes only. Not financial advice; past "
              "performance and model forecasts do not guarantee future results.",
              fontsize=7.4, color=pal.MUTED, style="italic")

    safe_name = "_".join(t.replace(".", "_") for t in tickers)
    png_path = os.path.join(SAVE_DIR, f"compare_{safe_name}.png")
    pdf_path = os.path.join(SAVE_DIR, f"compare_{safe_name}.pdf")
    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nComparison saved: {png_path}")
    print(f"Comparison saved: {pdf_path}")
    return png_path, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Side-by-side comparison of multiple tickers")
    parser.add_argument("--tickers", type=str, nargs="+", required=True)
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
    parser.add_argument("--risk_free_rate", type=float, default=0.04)
    parser.add_argument("--benchmark", type=str, default="^GSPC", dest="benchmark_ticker")
    parser.add_argument("--news_count", type=int, default=8)
    parser.add_argument("--skip_fundamentals", action="store_true")
    parser.add_argument("--skip_news", action="store_true")
    args = parser.parse_args()

    compare(
        args.tickers, lookback=args.lookback, pred_len=args.pred_len, range_=args.range_,
        interval=args.interval, tokenizer_name=args.tokenizer, model_name=args.model,
        device=args.device, T=args.T, top_p=args.top_p, ensemble_size=args.ensemble_size,
        sr_window=args.sr_window, sr_tolerance=args.sr_tolerance,
        risk_free_rate=args.risk_free_rate, benchmark_ticker=args.benchmark_ticker,
        news_count=args.news_count, skip_fundamentals=args.skip_fundamentals,
        skip_news=args.skip_news,
    )
