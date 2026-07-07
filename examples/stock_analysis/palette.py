# -*- coding: utf-8 -*-
"""Shared, validated color palette (light surface) used across every report page."""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
BLUE_300 = "#6da7ec"
BLUE_100 = "#cde2fb"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
MAGENTA = "#e87ba4"

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"

VERDICT_COLOR = {
    "BUY": GOOD, "BULLISH": GOOD,
    "SELL": CRITICAL, "BEARISH": CRITICAL,
    "HOLD": WARNING, "NEUTRAL / MIXED": WARNING,
    "positive": GOOD, "negative": CRITICAL, "neutral": WARNING,
}


def style_axes(ax):
    """Applies the shared chrome (surface, gridlines, muted spines/ticks) to a matplotlib Axes."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
