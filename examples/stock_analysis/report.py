# -*- coding: utf-8 -*-
"""
report.py

Renders the full analysis into a professional multi-page PDF, a quick-look
overview PNG (page 1), and a self-contained interactive HTML chart (Plotly).
Every plotting function takes the fully-computed `ctx` dict built by
stock_report.py — this module only visualizes, it does not compute anything.
"""

import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import palette as pal
from . import fundamental_analysis as fa

PAGE_SIZE = (12, 14.5)


def _new_page(title, subtitle):
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=pal.SURFACE)
    fig.text(0.06, 0.975, title, fontsize=18, fontweight="bold", color=pal.INK, ha="left", va="top")
    fig.text(0.06, 0.955, subtitle, fontsize=9.5, color=pal.MUTED, ha="left", va="top")
    return fig


def _verdict_banner(fig, rect, label, right_text):
    ax = fig.add_axes(rect)
    ax.axis("off")
    color = pal.VERDICT_COLOR.get(label, pal.MUTED)
    arrow = {"BUY": "▲", "BULLISH": "▲", "SELL": "▼", "BEARISH": "▼"}.get(label, "▬")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                 boxstyle="round,pad=0,rounding_size=0.04",
                                 linewidth=0, facecolor=color, alpha=0.14))
    ax.text(0.02, 0.5, f"{arrow}  {label}", transform=ax.transAxes, fontsize=16,
            fontweight="bold", color=color, va="center")
    ax.text(0.98, 0.5, right_text, transform=ax.transAxes, fontsize=11.5,
            color=pal.INK_SECONDARY, va="center", ha="right")


def _score_barh(ax, labels, values, title):
    y = np.arange(len(labels))
    colors = [pal.GOOD if v is not None and v >= 60 else
              (pal.CRITICAL if v is not None and v <= 40 else pal.WARNING) for v in values]
    plot_vals = [v if v is not None else 0 for v in values]
    bars = ax.barh(y, plot_vals, color=colors, height=0.55)
    for b, v in zip(bars, values):
        text = "n/a" if v is None else f"{v:.0f}"
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, text, va="center",
                fontsize=8.5, color=pal.INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.0, color=pal.INK_SECONDARY)
    ax.set_xlim(0, 108)
    ax.set_title(title, fontsize=10.5, color=pal.INK, loc="left", fontweight="bold")
    ax.axvline(50, color=pal.BASELINE, linewidth=0.8, linestyle=":")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.tick_params(length=0)


# --------------------------------------------------------------------------- Page 1
def build_page1_overview(ctx):
    fig = _new_page(
        f"{ctx['ticker']} — Complete Stock Analysis Report",
        f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} UTC  ·  Data through "
        f"{ctx['df']['date'].iloc[-1]:%Y-%m-%d}  ·  {ctx['interval']} bars  ·  "
        f"{len(ctx['close_paths'])}-run Kronos Monte-Carlo forecast",
    )
    _verdict_banner(fig, [0.06, 0.895, 0.90, 0.045], ctx["recommendation"],
                     f"Investment score: {ctx['investment_score']:.0f}/100"
                     if ctx["investment_score"] is not None else "Investment score: n/a")

    gs = GridSpec(2, 1, height_ratios=[3.4, 0.9], hspace=0.35, left=0.08, right=0.96,
                  top=0.87, bottom=0.06)
    tech = ctx["tech"]
    hist = ctx["df"].iloc[-ctx["lookback"]:].reset_index(drop=True)
    last_date = hist["date"].iloc[-1]
    close_paths = ctx["close_paths"]
    y_ts = ctx["y_timestamp"]
    mean_fc = close_paths.mean(axis=0)
    p5, p25, p75, p95 = np.percentile(close_paths, [5, 25, 75, 95], axis=0)

    ax_price = fig.add_subplot(gs[0])
    ax_price.plot(hist["date"], hist["close"], color=pal.INK, linewidth=2, label="Historical close")
    ax_price.plot(hist["date"], tech.ema50, color=pal.YELLOW, linewidth=1.3, label="EMA50")
    ax_price.plot(hist["date"], tech.ema200, color=pal.VIOLET, linewidth=1.3, label="EMA200")
    ax_price.plot(hist["date"], tech.bb_upper, color=pal.AQUA, linewidth=0.9, linestyle=":",
                  alpha=0.8, label="Bollinger 20,2")
    ax_price.plot(hist["date"], tech.bb_lower, color=pal.AQUA, linewidth=0.9, linestyle=":", alpha=0.8)

    ax_price.fill_between(y_ts, p5, p95, color=pal.BLUE_100, alpha=0.9, linewidth=0,
                          label="90% confidence interval")
    ax_price.fill_between(y_ts, p25, p75, color=pal.BLUE_300, alpha=0.85, linewidth=0,
                          label="50% confidence interval")
    ax_price.plot(y_ts, mean_fc, color=pal.BLUE, linewidth=2, linestyle="--",
                  label="Kronos forecast (ensemble mean)")

    level_prices = [p for p, _ in tech.support] + [p for p, _ in tech.resistance]
    all_vals = np.concatenate([
        hist["close"].values, tech.ema50.values, tech.ema200.values, p5, p95,
        np.array(level_prices) if level_prices else np.array([]),
    ])
    y_min, y_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    y_span = y_max - y_min
    ax_price.set_ylim(y_min - 0.05 * y_span, y_max + 0.24 * y_span)

    ax_price.axvline(last_date, color=pal.BASELINE, linewidth=1, linestyle=":")
    ax_price.text(last_date, y_max + 0.02 * y_span, " forecast start", fontsize=8,
                  color=pal.MUTED, va="bottom")

    for price, _ in tech.support:
        ax_price.axhline(price, color=pal.GOOD, linewidth=1, linestyle="--", alpha=0.7)
        ax_price.text(hist["date"].iloc[0], price, f" S ${price:,.2f}", fontsize=8, color=pal.GOOD,
                      va="bottom")
    for price, _ in tech.resistance:
        ax_price.axhline(price, color=pal.CRITICAL, linewidth=1, linestyle="--", alpha=0.7)
        ax_price.text(hist["date"].iloc[0], price, f" R ${price:,.2f}", fontsize=8,
                      color=pal.CRITICAL, va="bottom")

    cross_label = "Golden Cross" if tech.cross_regime == "golden" else "Death Cross"
    ax_price.text(0.99, 0.02, f"{cross_label} regime ({tech.bars_since_cross or '?'} bars ago)"
                  if tech.bars_since_cross else f"{cross_label} regime",
                  transform=ax_price.transAxes, ha="right", va="bottom", fontsize=8.5,
                  color=pal.INK_SECONDARY)

    ax_price.set_ylabel("Price", color=pal.INK_SECONDARY, fontsize=10)
    ax_price.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    pal.style_axes(ax_price)
    plt.setp(ax_price.get_xticklabels(), rotation=15, ha="right")

    ax_vol = fig.add_subplot(gs[1])
    ax_vol.bar(hist["date"], hist["volume"], color=pal.MUTED, width=1.6, alpha=0.6)
    ax_vol.axvline(last_date, color=pal.BASELINE, linewidth=1, linestyle=":")
    ax_vol.set_ylabel("Volume", color=pal.INK_SECONDARY, fontsize=9)
    pal.style_axes(ax_vol)
    plt.setp(ax_vol.get_xticklabels(), rotation=15, ha="right")

    return fig


# --------------------------------------------------------------------------- Page 2
def build_page2_technical(ctx):
    fig = _new_page(f"{ctx['ticker']} — Technical Analysis",
                     "RSI · MACD · ATR · ADX/DI · Golden/Death Cross")
    tech = ctx["tech"]
    hist = ctx["df"].iloc[-ctx["lookback"]:].reset_index(drop=True)

    gs = GridSpec(5, 1, height_ratios=[0.8, 0.8, 0.8, 0.8, 1.6], hspace=0.55, left=0.08,
                  right=0.96, top=0.90, bottom=0.05)

    ax_rsi = fig.add_subplot(gs[0])
    ax_rsi.plot(hist["date"], tech.rsi, color=pal.BLUE, linewidth=1.4)
    ax_rsi.axhline(70, color=pal.CRITICAL, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_rsi.axhline(30, color=pal.GOOD, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI(14)", fontsize=9, color=pal.INK_SECONDARY)
    pal.style_axes(ax_rsi)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    ax_macd = fig.add_subplot(gs[1], sharex=ax_rsi)
    hist_colors = np.where(tech.hist.values >= 0, pal.GOOD, pal.CRITICAL)
    ax_macd.bar(hist["date"], tech.hist, color=hist_colors, width=1.6, alpha=0.5)
    ax_macd.plot(hist["date"], tech.macd, color=pal.BLUE, linewidth=1.3, label="MACD")
    ax_macd.plot(hist["date"], tech.signal, color=pal.ORANGE, linewidth=1.3, label="Signal")
    ax_macd.axhline(0, color=pal.BASELINE, linewidth=0.8)
    ax_macd.set_ylabel("MACD", fontsize=9, color=pal.INK_SECONDARY)
    ax_macd.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    pal.style_axes(ax_macd)
    plt.setp(ax_macd.get_xticklabels(), visible=False)

    ax_atr = fig.add_subplot(gs[2], sharex=ax_rsi)
    atr_pct = tech.atr / hist["close"] * 100.0
    ax_atr.plot(hist["date"], atr_pct, color=pal.MAGENTA, linewidth=1.3)
    ax_atr.set_ylabel("ATR(14) %", fontsize=9, color=pal.INK_SECONDARY)
    pal.style_axes(ax_atr)
    plt.setp(ax_atr.get_xticklabels(), visible=False)

    ax_adx = fig.add_subplot(gs[3], sharex=ax_rsi)
    ax_adx.plot(hist["date"], tech.adx, color=pal.INK, linewidth=1.4, label="ADX")
    ax_adx.plot(hist["date"], tech.plus_di, color=pal.GOOD, linewidth=1.0, label="+DI")
    ax_adx.plot(hist["date"], tech.minus_di, color=pal.CRITICAL, linewidth=1.0, label="-DI")
    ax_adx.axhline(25, color=pal.BASELINE, linewidth=0.8, linestyle="--")
    ax_adx.set_ylabel("ADX/DI", fontsize=9, color=pal.INK_SECONDARY)
    ax_adx.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=3)
    pal.style_axes(ax_adx)
    plt.setp(ax_adx.get_xticklabels(), rotation=15, ha="right")

    ax_txt = fig.add_subplot(gs[4])
    ax_txt.axis("off")
    cross_label = "Golden Cross (bullish regime)" if tech.cross_regime == "golden" else \
        "Death Cross (bearish regime)"
    bb_width_last = tech.bb_width.iloc[-1]
    summary = (
        f"Cross regime:        {cross_label}"
        f"{f', {tech.bars_since_cross} bars ago' if tech.bars_since_cross else ''}\n"
        f"Trend strength (ADX): {tech.adx.iloc[-1]:.1f} ({tech.trend_strength}; ADX>=25 = trending)\n"
        f"RSI(14):              {tech.rsi.iloc[-1]:.1f} "
        f"({'overbought' if tech.rsi.iloc[-1] >= 70 else 'oversold' if tech.rsi.iloc[-1] <= 30 else 'neutral'})\n"
        f"MACD histogram:       {tech.hist.iloc[-1]:+.3f} "
        f"({'bullish' if tech.hist.iloc[-1] > 0 else 'bearish'} momentum)\n"
        f"ATR(14):              {tech.atr.iloc[-1]:.2f} ({tech.atr_pct:.1f}% of price)\n"
        f"Bollinger bandwidth:  {bb_width_last:.1f}% "
        f"({'squeeze / low volatility' if bb_width_last < 10 else 'expanded / high volatility' if bb_width_last > 25 else 'normal'})\n"
        f"Support levels:       {', '.join(f'${p:,.2f}' for p, _ in tech.support) or 'none identified'}\n"
        f"Resistance levels:    {', '.join(f'${p:,.2f}' for p, _ in tech.resistance) or 'none identified'}\n"
        f"Historical volatility: {tech.hist_vol_annualized:.1f}% (annualized)\n"
    )
    ax_txt.text(0.0, 1.0, "Technical summary", transform=ax_txt.transAxes, fontsize=11,
                fontweight="bold", color=pal.INK, va="top")
    # Escape literal "$" -- matplotlib treats unescaped "$...$" pairs as mathtext, which
    # silently drops the currency symbols and mangles spacing when a block has 2+ of them.
    ax_txt.text(0.0, 0.85, summary.replace("$", r"\$"), transform=ax_txt.transAxes, fontsize=9.3,
                color=pal.INK_SECONDARY, va="top", family="monospace", linespacing=1.7)
    return fig


# --------------------------------------------------------------------------- Page 3
def build_page3_fundamentals_valuation(ctx):
    fig = _new_page(f"{ctx['ticker']} — Fundamental Analysis & Valuation",
                     "Quality score · growth · cash flow · leverage · DCF & Graham fair value")
    fund = ctx["fund"]

    gs = GridSpec(3, 2, height_ratios=[1.5, 0.65, 1.4], hspace=0.45, wspace=0.32, left=0.16,
                  right=0.96, top=0.90, bottom=0.05)

    ax_quality = fig.add_subplot(gs[0, 0])
    if ctx["quality_score"] is not None:
        pillar_labels = list(ctx["quality_pillar_scores"].keys())
        pillar_vals = list(ctx["quality_pillar_scores"].values())
        _score_barh(ax_quality, [l.replace("_", " ").title() for l in pillar_labels], pillar_vals,
                    f"Quality Score: {ctx['quality_score']:.0f}/100")
    else:
        ax_quality.axis("off")
        ax_quality.text(0.5, 0.5, "Insufficient data for quality score", ha="center", va="center",
                         fontsize=9, color=pal.MUTED)

    ax_rev = fig.add_subplot(gs[0, 1])
    years = fund.get("yearly_history", [])
    if years:
        yrs = [y["year"] for y in years]
        revenue = [y["revenue"] for y in years]
        net_income = [y["net_income"] for y in years]
        x = np.arange(len(yrs))
        width = 0.36
        bars_rev = ax_rev.bar(x - width / 2, revenue, width, color=pal.BLUE, label="Revenue")
        bars_ni = ax_rev.bar(x + width / 2, net_income, width, color=pal.VIOLET, label="Net income")
        for bars in (bars_rev, bars_ni):
            for b in bars:
                h = b.get_height()
                ax_rev.text(b.get_x() + b.get_width() / 2, h, fa.fmt_money(h), rotation=90,
                            ha="center", va="bottom" if h >= 0 else "top", fontsize=6.6,
                            color=pal.INK_SECONDARY)
        ax_rev.set_xticks(x)
        ax_rev.set_xticklabels(yrs, fontsize=8.5, color=pal.MUTED)
        ax_rev.axhline(0, color=pal.BASELINE, linewidth=0.8)
        # Headroom for the rotated value labels above the tallest bars, so they don't run
        # into the legend; legend sits below the x-axis labels instead of inside the plot.
        ax_rev.set_ylim(top=max(revenue + net_income) * 1.22)
        ax_rev.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5,
                      frameon=False)
        ax_rev.set_yticklabels([])
        ax_rev.set_title("Revenue vs. net income (fiscal year)", fontsize=10, color=pal.INK,
                          loc="left", fontweight="bold")
        pal.style_axes(ax_rev)
    else:
        ax_rev.axis("off")
        ax_rev.text(0.5, 0.5, "No multi-year financials available", ha="center", va="center",
                     fontsize=9, color=pal.MUTED)

    ax_fund_txt = fig.add_subplot(gs[1, :])
    ax_fund_txt.axis("off")
    cur = fund.get("currency") or ""
    if fund.get("error"):
        ax_fund_txt.text(0.0, 1.0, fund["error"], transform=ax_fund_txt.transAxes, fontsize=9.5,
                          color=pal.MUTED, va="top")
    else:
        col1 = (
            f"Market cap:          {fa.fmt_money(fund['market_cap'], cur)}\n"
            f"P/E (trailing / fwd): {fa.fmt_ratio(fund['trailing_pe'])} / {fa.fmt_ratio(fund['forward_pe'])}\n"
            f"PEG ratio:           {fa.fmt_ratio(fund['peg_ratio'])}\n"
            f"P/S / P/B:           {fa.fmt_ratio(fund['price_to_sales'])} / {fa.fmt_ratio(fund['price_to_book'])}\n"
            f"EV / EBITDA:         {fa.fmt_ratio(fund['ev_to_ebitda'])}\n"
            f"Dividend yield:      {fa.fmt_pct(fund['dividend_yield'])}   Payout: {fa.fmt_pct0(fund['payout_ratio'])}\n"
        )
        col2 = (
            f"Revenue growth YoY:   {fa.fmt_pct(fund['revenue_growth_yoy'])}\n"
            f"Earnings growth YoY:  {fa.fmt_pct(fund['earnings_growth_yoy'])}\n"
            f"Revenue CAGR (hist.): {fa.fmt_pct(fund.get('revenue_cagr'))}\n"
            f"Gross/Op/Net margin:  {fa.fmt_pct0(fund['gross_margin'])}/"
            f"{fa.fmt_pct0(fund['operating_margin'])}/{fa.fmt_pct0(fund['profit_margin'])}\n"
            f"ROE / ROIC (approx.): {fa.fmt_pct(fund['return_on_equity'])} / {fa.fmt_pct(fund.get('roic'))}\n"
            f"ROA:                  {fa.fmt_pct(fund['return_on_assets'])}\n"
        )
        col3 = (
            f"Free cash flow:      {fa.fmt_money(fund['free_cash_flow'], cur)}\n"
            f"Operating cash flow: {fa.fmt_money(fund['operating_cash_flow'], cur)}\n"
            f"Total debt / cash:   {fa.fmt_money(fund['total_debt'], cur)} / {fa.fmt_money(fund['total_cash'], cur)}\n"
            f"Net debt:            {fa.fmt_money(fund['net_debt'], cur)}\n"
            f"Debt / Equity:       {fa.fmt_ratio(fund['debt_to_equity'])}\n"
            f"Current / Quick:     {fa.fmt_ratio(fund['current_ratio'])} / {fa.fmt_ratio(fund['quick_ratio'])}\n"
        )
        ax_fund_txt.text(0.0, 1.0, "Valuation", transform=ax_fund_txt.transAxes, fontsize=10.5,
                          fontweight="bold", color=pal.INK, va="top")
        ax_fund_txt.text(0.0, 0.86, col1, transform=ax_fund_txt.transAxes, fontsize=8.4,
                          color=pal.INK_SECONDARY, va="top", family="monospace", linespacing=1.7)
        ax_fund_txt.text(0.35, 1.0, "Growth & profitability", transform=ax_fund_txt.transAxes,
                          fontsize=10.5, fontweight="bold", color=pal.INK, va="top")
        ax_fund_txt.text(0.35, 0.86, col2, transform=ax_fund_txt.transAxes, fontsize=8.4,
                          color=pal.INK_SECONDARY, va="top", family="monospace", linespacing=1.7)
        ax_fund_txt.text(0.70, 1.0, "Cash flow & leverage", transform=ax_fund_txt.transAxes,
                          fontsize=10.5, fontweight="bold", color=pal.INK, va="top")
        ax_fund_txt.text(0.70, 0.86, col3, transform=ax_fund_txt.transAxes, fontsize=8.4,
                          color=pal.INK_SECONDARY, va="top", family="monospace", linespacing=1.7)

    ax_val = fig.add_subplot(gs[2, 0])
    labels, vals, colors = [], [], []
    price = ctx["last_close"]
    labels.append("Current price"); vals.append(price); colors.append(pal.INK)
    if ctx["dcf"] is not None:
        labels.append("DCF fair value"); vals.append(ctx["dcf"]["fair_value"]); colors.append(pal.BLUE)
    if ctx["graham"] is not None:
        labels.append("Graham number"); vals.append(ctx["graham"]); colors.append(pal.AQUA)
    if ctx.get("target_mean_price"):
        labels.append("Analyst target"); vals.append(ctx["target_mean_price"]); colors.append(pal.ORANGE)
    y = np.arange(len(labels))
    bars = ax_val.barh(y, vals, color=colors, height=0.5)
    for b, v in zip(bars, vals):
        ax_val.text(b.get_width(), b.get_y() + b.get_height() / 2, f" ${v:,.2f}", va="center",
                    fontsize=8.5, color=pal.INK_SECONDARY)
    ax_val.set_yticks(y)
    ax_val.set_yticklabels(labels, fontsize=9, color=pal.INK_SECONDARY)
    ax_val.invert_yaxis()
    ax_val.set_title("Price vs. estimated fair value", fontsize=10, color=pal.INK, loc="left",
                      fontweight="bold")
    for spine in ax_val.spines.values():
        spine.set_visible(False)
    ax_val.set_xticks([])
    ax_val.tick_params(length=0)

    ax_val_txt = fig.add_subplot(gs[2, 1])
    ax_val_txt.axis("off")
    dcf_line = "n/a — FCF negative/unavailable"
    assumptions = ""
    if ctx["dcf"] is not None:
        d = ctx["dcf"]
        dcf_line = f"${d['fair_value']:,.2f}"
        assumptions = (
            f"DCF assumptions: discount rate {d['discount_rate'] * 100:.1f}% (CAPM-style), "
            f"Y1 FCF growth {d['growth_y1'] * 100:.1f}% fading to a "
            f"{d['terminal_growth'] * 100:.1f}% terminal growth over {d['years']} years."
        )
    graham_line = f"${ctx['graham']:,.2f}" if ctx["graham"] is not None else "n/a — EPS/book value <= 0"
    mos_dcf = ctx.get("margin_of_safety_dcf")
    mos_graham = ctx.get("margin_of_safety_graham")
    txt = (
        f"DCF fair value:            {dcf_line}\n"
        f"Graham number:             {graham_line}\n"
        f"Blended fair value:        "
        f"{'$%.2f' % ctx['blended_fair_value'] if ctx['blended_fair_value'] is not None else 'n/a'}\n"
        f"Margin of safety (DCF):    {fa.fmt_pct(mos_dcf)}\n"
        f"Margin of safety (Graham): {fa.fmt_pct(mos_graham)}\n"
    )
    ax_val_txt.text(0.0, 1.0, "Margin of safety", transform=ax_val_txt.transAxes, fontsize=10.5,
                     fontweight="bold", color=pal.INK, va="top")
    ax_val_txt.text(0.0, 0.84, txt.replace("$", r"\$"), transform=ax_val_txt.transAxes, fontsize=8.6,
                     color=pal.INK_SECONDARY, va="top", family="monospace", linespacing=1.8)
    ax_val_txt.text(0.0, 0.34, assumptions, transform=ax_val_txt.transAxes, fontsize=7.4,
                     color=pal.MUTED, va="top", style="italic", wrap=True)
    ax_val_txt.text(0.0, 0.14,
                     "DCF and Graham are simplified, transparent models, not precise fair-value "
                     "guarantees.", transform=ax_val_txt.transAxes, fontsize=7.4, color=pal.MUTED,
                     va="top", style="italic", wrap=True)
    return fig


def _render_bullets(ax, x, y0, items, char_width=58, fontsize=7.6, color=None, line_step=0.038,
                     max_lines=6):
    """Word-wraps each bullet to `char_width` chars (so it can't bleed into a neighboring
    column) and stacks the wrapped lines, indenting continuation lines under the marker.
    Returns the y position after the last line drawn."""
    color = color or pal.INK_SECONDARY
    y = y0
    lines_used = 0
    for item in items:
        wrapped = textwrap.wrap(item, width=char_width) or [""]
        for j, line in enumerate(wrapped):
            if lines_used >= max_lines:
                return y
            prefix = "•  " if j == 0 else "   "
            ax.text(x, y, f"{prefix}{line}", transform=ax.transAxes, fontsize=fontsize,
                    color=color, va="top")
            y -= line_step
            lines_used += 1
    return y


# --------------------------------------------------------------------------- Page 4
def build_page4_risk_news_quality(ctx):
    fig = _new_page(f"{ctx['ticker']} — Risk, News Sentiment & Company Quality",
                     "Volatility · beta · drawdown · Sharpe/Sortino · recent news · moat signals")
    risk = ctx["risk"]
    news = ctx["news"]
    qual = ctx["quality_assessment"]
    hist = ctx["df"].iloc[-ctx["lookback"]:].reset_index(drop=True)

    gs = GridSpec(3, 2, height_ratios=[1.2, 1.15, 1.9], hspace=0.4, wspace=0.28, left=0.08,
                  right=0.96, top=0.90, bottom=0.04)

    ax_dd = fig.add_subplot(gs[0, 0])
    running_max = hist["close"].cummax()
    drawdown = (hist["close"] / running_max - 1.0) * 100.0
    ax_dd.fill_between(hist["date"], drawdown, 0, color=pal.CRITICAL, alpha=0.25, linewidth=0)
    ax_dd.plot(hist["date"], drawdown, color=pal.CRITICAL, linewidth=1.2)
    ax_dd.set_title("Drawdown from running peak", fontsize=10, color=pal.INK, loc="left",
                     fontweight="bold")
    ax_dd.set_ylabel("%", fontsize=9, color=pal.INK_SECONDARY)
    pal.style_axes(ax_dd)
    plt.setp(ax_dd.get_xticklabels(), rotation=15, ha="right", fontsize=7.5)

    ax_risk_txt = fig.add_subplot(gs[0, 1])
    ax_risk_txt.axis("off")
    risk_txt = (
        f"Historical volatility (ann.): {ctx['tech'].hist_vol_annualized:.1f}%\n"
        f"Beta (computed vs S&P 500):   {fa.fmt_ratio(risk.beta)}\n"
        f"Maximum drawdown:             {risk.max_drawdown_pct:.1f}% "
        f"({risk.drawdown_peak_date:%Y-%m-%d} -> {risk.drawdown_trough_date:%Y-%m-%d})\n"
        f"Sharpe ratio (rf={risk.risk_free_annual * 100:.1f}%): {fa.fmt_ratio(risk.sharpe)}\n"
        f"Sortino ratio (rf={risk.risk_free_annual * 100:.1f}%):{fa.fmt_ratio(risk.sortino)}\n"
    )
    ax_risk_txt.text(0.0, 1.0, "Risk metrics", transform=ax_risk_txt.transAxes, fontsize=10.5,
                      fontweight="bold", color=pal.INK, va="top")
    ax_risk_txt.text(0.0, 0.82, risk_txt, transform=ax_risk_txt.transAxes, fontsize=9.2,
                      color=pal.INK_SECONDARY, va="top", family="monospace", linespacing=1.9)

    ax_news = fig.add_subplot(gs[1, :])
    ax_news.axis("off")
    overall = news["overall_label"]
    ax_news.text(0.0, 1.0, "Recent news & sentiment", transform=ax_news.transAxes, fontsize=10.5,
                 fontweight="bold", color=pal.INK, va="top")
    ax_news.text(1.0, 1.0, f"Overall: {overall.upper()}  "
                 f"(+{news['counts']['positive']} / ={news['counts']['neutral']} / "
                 f"-{news['counts']['negative']})",
                 transform=ax_news.transAxes, fontsize=9.5, color=pal.VERDICT_COLOR.get(overall, pal.MUTED),
                 va="top", ha="right", fontweight="bold")
    y = 0.86
    for item in news["items"][:6]:
        color = pal.VERDICT_COLOR.get(item["sentiment_label"], pal.MUTED)
        date_str = f"{item['published']:%Y-%m-%d}" if item.get("published") else ""
        title = textwrap.shorten(item["title"], width=100, placeholder="...").replace("$", r"\$")
        ax_news.text(0.0, y, "●", transform=ax_news.transAxes, fontsize=9, color=color, va="top")
        ax_news.text(0.02, y, f"{title}", transform=ax_news.transAxes, fontsize=8.6,
                     color=pal.INK_SECONDARY, va="top")
        ax_news.text(1.0, y, f"{item['publisher']}  {date_str}", transform=ax_news.transAxes,
                     fontsize=7.6, color=pal.MUTED, va="top", ha="right")
        y -= 0.135
    if not news["items"]:
        ax_news.text(0.0, 0.86, "No recent news available.", transform=ax_news.transAxes,
                     fontsize=9, color=pal.MUTED, va="top")

    ax_qual = fig.add_subplot(gs[2, :])
    ax_qual.axis("off")
    biz_lines = textwrap.wrap(
        (qual.get("business_summary") or "n/a").replace("$", r"\$"), width=148)[:3]
    ax_qual.text(0.0, 1.0, "Company quality", transform=ax_qual.transAxes, fontsize=10.5,
                 fontweight="bold", color=pal.INK, va="top")
    y_biz = 0.94
    for line in biz_lines:
        ax_qual.text(0.0, y_biz, line, transform=ax_qual.transAxes, fontsize=8.2,
                     color=pal.INK_SECONDARY, va="top")
        y_biz -= 0.045
    section_top = y_biz - 0.03

    ax_qual.text(0.0, section_top, "Market position", transform=ax_qual.transAxes, fontsize=9.3,
                 fontweight="bold", color=pal.INK, va="top")
    y = _render_bullets(ax_qual, 0.0, section_top - 0.055, qual["market_position"][:3])
    ax_qual.text(0.0, y - 0.02, "Moat signals (quantitative proxy)", transform=ax_qual.transAxes,
                 fontsize=9.3, fontweight="bold", color=pal.INK, va="top")
    _render_bullets(ax_qual, 0.0, y - 0.075, qual["moat_signals"][:2], max_lines=4)

    ax_qual.text(0.52, section_top, "Growth opportunities", transform=ax_qual.transAxes,
                 fontsize=9.3, fontweight="bold", color=pal.INK, va="top")
    y2 = _render_bullets(ax_qual, 0.52, section_top - 0.055, qual["growth_opportunities"][:2],
                          max_lines=4)
    ax_qual.text(0.52, y2 - 0.02, "Key risks", transform=ax_qual.transAxes, fontsize=9.3,
                 fontweight="bold", color=pal.CRITICAL, va="top")
    _render_bullets(ax_qual, 0.52, y2 - 0.075, qual["key_risks"][:3], max_lines=5)

    ax_qual.text(0.0, 0.02, qual["caveat"], transform=ax_qual.transAxes, fontsize=7.2,
                 color=pal.MUTED, va="bottom", style="italic", wrap=True)
    return fig


# --------------------------------------------------------------------------- Page 5
def build_page5_final_score(ctx):
    fig = _new_page(f"{ctx['ticker']} — Overall Rating", "Blended Investment Score & recommendation")
    _verdict_banner(fig, [0.06, 0.87, 0.90, 0.06], ctx["recommendation"],
                     f"Investment score: {ctx['investment_score']:.0f}/100"
                     if ctx["investment_score"] is not None else "n/a")

    gs = GridSpec(2, 1, height_ratios=[0.65, 1.25], hspace=0.18, left=0.26, right=0.90, top=0.78,
                  bottom=0.10)
    ax_score = fig.add_subplot(gs[0])
    labels = list(ctx["score_breakdown"].keys())
    vals = list(ctx["score_breakdown"].values())
    _score_barh(ax_score, labels, vals, "Investment score components")

    ax_reasons = fig.add_subplot(gs[1])
    ax_reasons.axis("off")
    ax_reasons.text(0.0, 1.0, "Reasoning", transform=ax_reasons.transAxes, fontsize=12,
                     fontweight="bold", color=pal.INK, va="top")
    y = 0.90
    for reason in ctx["reasons"]:
        ax_reasons.text(0.0, y, f"•  {reason}", transform=ax_reasons.transAxes, fontsize=9.6,
                         color=pal.INK_SECONDARY, va="top", wrap=True)
        y -= 0.095

    weights_txt = "  ·  ".join(f"{k} {v * 100:.0f}%" for k, v in ctx["score_weights"].items())
    ax_reasons.text(0.0, y - 0.03, f"Fixed weights: {weights_txt}", transform=ax_reasons.transAxes,
                     fontsize=8, color=pal.MUTED, va="top", wrap=True)
    ax_reasons.text(0.0, y - 0.10,
                     "Disclaimer: generated by a machine-learning model (Kronos) and rule-based "
                     "financial heuristics for research/educational purposes only. Not financial "
                     "advice; past performance and model forecasts do not guarantee future results.",
                     transform=ax_reasons.transAxes, fontsize=7.6, color=pal.MUTED, va="top",
                     style="italic", wrap=True)
    return fig


def build_pdf_and_png(ctx, pdf_path, png_path):
    pages = [
        build_page1_overview(ctx),
        build_page2_technical(ctx),
        build_page3_fundamentals_valuation(ctx),
        build_page4_risk_news_quality(ctx),
        build_page5_final_score(ctx),
    ]
    with PdfPages(pdf_path) as pdf:
        for fig in pages:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
    pages[0].savefig(png_path, dpi=150, facecolor=pages[0].get_facecolor())
    for fig in pages:
        plt.close(fig)


# --------------------------------------------------------------------------- Interactive HTML
def build_interactive_html(ctx, html_path):
    tech = ctx["tech"]
    hist = ctx["df"].iloc[-ctx["lookback"]:].reset_index(drop=True)
    close_paths = ctx["close_paths"]
    y_ts = pd.Series(ctx["y_timestamp"])
    mean_fc = close_paths.mean(axis=0)
    p5, p95 = np.percentile(close_paths, [5, 95], axis=0)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.14, 0.18, 0.18], vertical_spacing=0.03,
        subplot_titles=("Price, forecast & confidence interval", "Volume", "RSI(14)", "MACD"),
    )

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], name="Historical close",
                              line=dict(color=pal.INK, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["date"], y=tech.ema50, name="EMA50",
                              line=dict(color=pal.YELLOW, width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["date"], y=tech.ema200, name="EMA200",
                              line=dict(color=pal.VIOLET, width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(y_ts) + list(y_ts[::-1]), y=list(p95) + list(p5[::-1]),
                              fill="toself", fillcolor="rgba(42,120,214,0.15)",
                              line=dict(color="rgba(0,0,0,0)"), name="90% CI", hoverinfo="skip"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=y_ts, y=mean_fc, name="Kronos forecast (mean)",
                              line=dict(color=pal.BLUE, width=2, dash="dash")), row=1, col=1)

    fig.add_trace(go.Bar(x=hist["date"], y=hist["volume"], name="Volume",
                          marker_color=pal.MUTED, opacity=0.6), row=2, col=1)

    fig.add_trace(go.Scatter(x=hist["date"], y=tech.rsi, name="RSI(14)",
                              line=dict(color=pal.BLUE, width=1.4)), row=3, col=1)
    fig.add_hline(y=70, line=dict(color=pal.CRITICAL, dash="dash", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=pal.GOOD, dash="dash", width=1), row=3, col=1)

    fig.add_trace(go.Bar(x=hist["date"], y=tech.hist, name="MACD histogram",
                          marker_color=np.where(tech.hist.values >= 0, pal.GOOD, pal.CRITICAL),
                          opacity=0.5), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist["date"], y=tech.macd, name="MACD",
                              line=dict(color=pal.BLUE, width=1.3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist["date"], y=tech.signal, name="Signal",
                              line=dict(color=pal.ORANGE, width=1.3)), row=4, col=1)

    fig.update_layout(
        title=f"{ctx['ticker']} — Interactive Kronos Stock Analysis "
              f"({ctx['recommendation']}, score {ctx['investment_score']:.0f}/100)"
              if ctx["investment_score"] is not None else f"{ctx['ticker']} — Interactive Analysis",
        template="plotly_white", height=1000, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=90, b=40),
        plot_bgcolor=pal.SURFACE, paper_bgcolor=pal.SURFACE,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
