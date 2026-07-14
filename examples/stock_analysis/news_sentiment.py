# -*- coding: utf-8 -*-
"""
news_sentiment.py

Fetches recent company news (via data_provider.fetch_news) and scores each
headline/summary with VADER (a lexicon + rule-based sentiment analyzer;
lightweight, deterministic, no model download). Aggregates into an overall
positive/neutral/negative sentiment label for the report.
"""

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def _label(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def analyze_news(news_items: list) -> dict:
    """
    Scores each {title, summary, ...} item with VADER and returns:
      {items: [...with sentiment_compound/sentiment_label...],
       avg_compound, overall_label, counts: {positive, neutral, negative}}
    """
    scored = []
    for item in news_items:
        text = f"{item.get('title', '')}. {item.get('summary', '')}".strip()
        compound = _analyzer.polarity_scores(text)["compound"] if text else 0.0
        scored.append({**item, "sentiment_compound": compound, "sentiment_label": _label(compound)})

    if scored:
        avg_compound = float(np.mean([s["sentiment_compound"] for s in scored]))
    else:
        avg_compound = 0.0

    counts = {
        "positive": sum(1 for s in scored if s["sentiment_label"] == "positive"),
        "neutral": sum(1 for s in scored if s["sentiment_label"] == "neutral"),
        "negative": sum(1 for s in scored if s["sentiment_label"] == "negative"),
    }

    return {
        "items": sorted(scored, key=lambda s: s.get("published") or 0, reverse=True),
        "avg_compound": avg_compound,
        "overall_label": _label(avg_compound),
        "counts": counts,
    }
