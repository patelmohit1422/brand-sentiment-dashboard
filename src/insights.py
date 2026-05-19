from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from .config import INSIGHTS_PATH


PRICE_WORDS = {"pricing", "price", "cost", "expensive", "cheap", "billing", "refund", "refunds"}
SPEED_WORDS = {"slow", "delay", "delayed", "delivery", "waiting", "late", "lag", "loading"}
SUPPORT_WORDS = {"support", "help", "response", "responded", "customer", "care", "team"}
BUG_WORDS = {"bug", "bugs", "broken", "glitch", "glitchy", "crash", "crashes", "frozen", "error"}
PRAISE_WORDS = {"fast", "smooth", "easy", "helpful", "great", "good", "clean", "reliable", "nice", "solid"}


def _count_keyword_hits(texts, keywords):
    total = 0
    for text in texts:
        words = set(str(text).lower().split())
        total += len(words & keywords)
    return total


def build_insights(df: pd.DataFrame, metrics: dict, top_keywords: dict) -> dict:
    total = len(df)
    counts = df["label"].value_counts().to_dict()
    dominant = max(counts, key=counts.get)
    dominant_pct = round(counts[dominant] / total * 100, 1)

    neg_texts = df.loc[df["label"] == "Negative", "text"].astype(str).tolist()
    pos_texts = df.loc[df["label"] == "Positive", "text"].astype(str).tolist()
    neu_texts = df.loc[df["label"] == "Neutral", "text"].astype(str).tolist()

    complaint_theme_counts = {
        "pricing": _count_keyword_hits(neg_texts, PRICE_WORDS),
        "delays": _count_keyword_hits(neg_texts, SPEED_WORDS),
        "support": _count_keyword_hits(neg_texts, SUPPORT_WORDS),
        "bugs": _count_keyword_hits(neg_texts, BUG_WORDS),
    }
    praise_theme_counts = {
        "speed": _count_keyword_hits(pos_texts, {"fast", "quick", "smooth"}),
        "ease": _count_keyword_hits(pos_texts, {"easy", "simple", "clean"}),
        "helpfulness": _count_keyword_hits(pos_texts, {"helpful", "support", "friendly"}),
    }

    top_pos_terms = []
    top_neg_terms = []
    for cls, terms in top_keywords.items():
        if cls == "Positive":
            top_pos_terms = [x["term"] for x in terms.get("top_positive_terms", [])[:5]]
        elif cls == "Negative":
            top_neg_terms = [x["term"] for x in terms.get("top_positive_terms", [])[:5]]

    if counts.get("Negative", 0) > counts.get("Positive", 0):
        suggestion = "The product is drawing more complaints than praise, so the next move should be to fix friction points before pushing new features."
    elif counts.get("Positive", 0) >= counts.get("Negative", 0):
        suggestion = "The audience is leaning positive, so the team should protect the current experience and turn the best feedback into marketing proof."
    else:
        suggestion = "The feedback is mixed, so the team should separate product problems from UX confusion and handle the biggest repeated issues first."

    if complaint_theme_counts["pricing"] >= complaint_theme_counts["bugs"] and complaint_theme_counts["pricing"] >= complaint_theme_counts["delays"]:
        personal_insight = "I noticed that a lot of the negative feedback is tied to pricing and value perception, which tells me people are not just reacting to the product, they are judging whether it feels worth the money."
    elif complaint_theme_counts["delays"] >= complaint_theme_counts["bugs"]:
        personal_insight = "I noticed that delays and slow responses show up a lot in the negative comments, so speed seems to matter more here than fancy features."
    else:
        personal_insight = "I noticed that support and bug-related complaints keep appearing together, which makes the experience feel less like a small issue and more like a trust problem."

    key_observations = [
        f"{dominant} sentiment is the largest segment at {dominant_pct}%.",
        f"Positive comments often mention words like {', '.join(top_pos_terms[:3]) if top_pos_terms else 'fast and easy'}.",
        f"Negative comments keep circling back to {', '.join([k for k, v in complaint_theme_counts.items() if v == max(complaint_theme_counts.values())][:2])}.",
        f"Neutral comments are still a big chunk, which usually means users are observing more than reacting.",
    ]

    insights = {
        "summary": f"{dominant} sentiment is the largest segment at {dominant_pct}%.",
        "distribution": counts,
        "model_quality": {
            "accuracy": metrics["accuracy"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        },
        "top_positive_terms": top_pos_terms,
        "top_negative_terms": top_neg_terms,
        "suggestion": suggestion,
        "key_observations": key_observations,
        "complaint_themes": complaint_theme_counts,
        "praise_themes": praise_theme_counts,
        "personal_insight": personal_insight,
        "sample_counts": {"positive": len(pos_texts), "negative": len(neg_texts), "neutral": len(neu_texts)},
    }
    INSIGHTS_PATH.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    return insights


def load_insights():
    if INSIGHTS_PATH.exists():
        return json.loads(INSIGHTS_PATH.read_text(encoding="utf-8"))
    return {}
