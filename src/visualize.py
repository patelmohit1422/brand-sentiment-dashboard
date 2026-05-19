from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from .config import FIGURES_DIR, TOP_KEYWORDS_PATH
from .text_processing import preprocess_text


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def create_visualizations(df: pd.DataFrame, top_keywords: dict | None = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Sentiment distribution
    counts = df["label"].value_counts().reindex(["Positive", "Negative", "Neutral"]).fillna(0)
    plt.figure(figsize=(8.5, 5))
    bars = plt.bar(counts.index, counts.values)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.2)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom")
    _savefig(FIGURES_DIR / "sentiment_distribution.png")

    # Weekly sentiment trend
    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    trend = trend_df.groupby([pd.Grouper(key="date", freq="W"), "label"]).size().reset_index(name="count")
    plt.figure(figsize=(10, 5))
    for label in ["Positive", "Negative", "Neutral"]:
        subset = trend[trend["label"] == label]
        plt.plot(subset["date"], subset["count"], marker="o", linewidth=2, label=label)
    plt.title("Weekly Sentiment Trend")
    plt.xlabel("Week")
    plt.ylabel("Mentions")
    plt.grid(alpha=0.2)
    plt.xticks(rotation=20)
    plt.legend()
    _savefig(FIGURES_DIR / "sentiment_trend.png")

    # Word clouds per class
    for label in ["Positive", "Negative", "Neutral"]:
        text = " ".join(df.loc[df["label"] == label, "text"].astype(str).tolist())
        cleaned = preprocess_text(text) or "sentiment"
        wc = WordCloud(width=1200, height=700, background_color="white", collocations=False).generate(cleaned)
        plt.figure(figsize=(11, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{label} Word Cloud")
        _savefig(FIGURES_DIR / f"wordcloud_{label.lower()}.png")

    # Top keyword charts
    if top_keywords is None and TOP_KEYWORDS_PATH.exists():
        top_keywords = json.loads(TOP_KEYWORDS_PATH.read_text(encoding="utf-8"))

    if top_keywords:
        for sentiment, vals in top_keywords.items():
            terms = vals.get("top_positive_terms", [])[:10]
            if not terms:
                continue
            subset = pd.DataFrame(terms).sort_values("weight")
            plt.figure(figsize=(8.5, 5))
            plt.barh(subset["term"], subset["weight"])
            plt.title(f"Top Keywords - {sentiment}")
            plt.xlabel("Coefficient Weight")
            plt.ylabel("Term")
            plt.grid(axis="x", alpha=0.2)
            _savefig(FIGURES_DIR / f"top_keywords_{sentiment.lower()}.png")

    # A small extra chart for product thinking.
    source_counts = df["source"].value_counts()
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(source_counts.index, source_counts.values)
    plt.title("Source Mix")
    plt.xlabel("Source")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.2)
    _savefig(FIGURES_DIR / "source_mix.png")


if __name__ == "__main__":
    from .train_model import train_and_save_model

    df, metrics, top_keywords = train_and_save_model()
    create_visualizations(df, top_keywords)
    print("Visualizations created.")
