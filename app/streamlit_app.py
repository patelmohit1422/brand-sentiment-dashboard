from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, METRICS_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, MODEL_PATH, TOP_KEYWORDS_PATH
from src.train_model import train_and_save_model, predict_text
from src.visualize import create_visualizations
from src.insights import build_insights, load_insights

st.set_page_config(page_title="Brand Sentiment Analysis Dashboard", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_project_data():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not PROCESSED_DATA_PATH.exists() or not TOP_KEYWORDS_PATH.exists():
        df, metrics, top_keywords = train_and_save_model(force_regenerate_data=not RAW_DATA_PATH.exists())
        create_visualizations(df, top_keywords)
        insights = build_insights(df, metrics, top_keywords)
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        top_keywords = json.loads(TOP_KEYWORDS_PATH.read_text(encoding="utf-8")) if TOP_KEYWORDS_PATH.exists() else {}
        insights = load_insights() or build_insights(df, metrics, top_keywords)
    return df, metrics, top_keywords, insights


def show_metric_cards(metrics):
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
    cols[1].metric("Precision", f"{metrics['precision_weighted']:.2%}")
    cols[2].metric("Recall", f"{metrics['recall_weighted']:.2%}")
    cols[3].metric("F1 Score", f"{metrics['f1_weighted']:.2%}")


def display_saved_figure(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def _prediction_summary(sentiment: str, probs: dict[str, float]):
    ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = ordered[0]

    st.markdown("### Prediction Result")
    badge = {
        "Positive": "🟢 Positive",
        "Negative": "🔴 Negative",
        "Neutral": "🟡 Neutral",
    }.get(sentiment, sentiment)
    st.markdown(f"**Sentiment:** {badge}")
    st.metric("Confidence", f"{top_prob:.1%}")

    prob_df = pd.DataFrame(ordered, columns=["Sentiment", "Probability"])
    fig = px.bar(
        prob_df,
        x="Sentiment",
        y="Probability",
        text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
        title="Class Probability Breakdown",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 1], height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.write("Top class probabilities:")
    for label, prob in ordered:
        st.progress(float(prob), text=f"{label}: {prob:.1%}")


def app():
    df, metrics, top_keywords, insights = load_project_data()

    st.markdown("# Brand Sentiment Analysis Dashboard")
    st.caption("A portfolio-style NLP project that turns brand comments into sentiment predictions, charts, and business insights.")

    with st.sidebar:
        st.header("Project Summary")
        st.write(f"Samples: **{len(df):,}**")
        st.write("Classes: Positive / Negative / Neutral")
        st.write("Model: Logistic Regression + TF-IDF")
        st.write("Dataset: Synthetic brand-related social text")
        st.divider()
        st.subheader("Latest Insights")
        st.write(insights.get("summary", "Insights will appear here after training."))
        st.write(insights.get("personal_insight", insights.get("suggestion", "")))

    tabs = st.tabs(["Predict", "Dashboard", "Insights", "Data"])

    with tabs[0]:
        st.subheader("Predict Sentiment")
        st.write("Paste a tweet, review, or customer comment below. The model will clean the text, predict the sentiment, and show the confidence for each class.")
        demo_col1, demo_col2 = st.columns([1.15, 0.85])
        with demo_col1:
            user_text = st.text_area(
                "Enter text",
                placeholder="Example: The new update is smooth and the support team responded fast.",
                height=180,
            )
            run_btn = st.button("Analyze Sentiment", type="primary")
            st.caption("Try a short, messy, real-world sentence instead of perfect grammar. That is closer to how people actually write.")

        with demo_col2:
            st.markdown("### What the model returns")
            st.info("A predicted label, the probability for each class, and the cleaned version of your text.")
            st.markdown("### Sample inputs")
            st.write("• I love how smooth the checkout feels now.")
            st.write("• The app keeps crashing after the update.")
            st.write("• It is fine overall, just nothing special.")

        if run_btn and user_text.strip():
            sentiment, probs, cleaned = predict_text(user_text)
            left, right = st.columns([1.1, 0.9])
            with left:
                _prediction_summary(sentiment, probs)
            with right:
                st.markdown("### Cleaned text")
                st.code(cleaned or "[empty after cleaning]", language="text")
                st.markdown("### Model note")
                st.write("The TF-IDF vectorizer turns the cleaned text into numeric features before Logistic Regression scores the three sentiment classes.")
        elif run_btn:
            st.warning("Please enter some text before running the prediction.")

    with tabs[1]:
        st.subheader("Dashboard")
        st.write("This section is the visual proof that the project is not just a classifier. It shows the sentiment pattern, the trend over time, and the words that drive each class.")
        show_metric_cards(metrics)

        c1, c2 = st.columns(2)
        with c1:
            display_saved_figure(FIGURES_DIR / "sentiment_distribution.png", "Sentiment distribution")
        with c2:
            display_saved_figure(FIGURES_DIR / "sentiment_trend.png", "Weekly sentiment trend")

        st.markdown("### Word Clouds")
        w1, w2, w3 = st.columns(3)
        with w1:
            display_saved_figure(FIGURES_DIR / "wordcloud_positive.png", "Positive word cloud")
        with w2:
            display_saved_figure(FIGURES_DIR / "wordcloud_negative.png", "Negative word cloud")
        with w3:
            display_saved_figure(FIGURES_DIR / "wordcloud_neutral.png", "Neutral word cloud")

        st.markdown("### Top Keyword Charts")
        k1, k2, k3 = st.columns(3)
        with k1:
            display_saved_figure(FIGURES_DIR / "top_keywords_positive.png", "Top positive keywords")
        with k2:
            display_saved_figure(FIGURES_DIR / "top_keywords_negative.png", "Top negative keywords")
        with k3:
            display_saved_figure(FIGURES_DIR / "top_keywords_neutral.png", "Top neutral keywords")

        st.markdown("### Source Mix")
        display_saved_figure(FIGURES_DIR / "source_mix.png", "Source mix across the dataset")

    with tabs[2]:
        st.subheader("Business Insights")
        left, right = st.columns(2)
        with left:
            st.markdown("#### What users feel")
            st.write(insights.get("summary", "No insight available."))
            st.markdown("#### My personal read")
            st.write(insights.get("personal_insight", ""))
            st.markdown("#### Suggested action")
            st.write(insights.get("suggestion", ""))
        with right:
            st.markdown("#### Model quality")
            st.json(insights.get("model_quality", metrics))
            st.markdown("#### Complaint themes")
            st.json(insights.get("complaint_themes", {}))
            st.markdown("#### Praise themes")
            st.json(insights.get("praise_themes", {}))

        st.markdown("### Key observations")
        for item in insights.get("key_observations", []):
            st.write(f"- {item}")

        if top_keywords:
            st.markdown("### Top terms by sentiment")
            for sentiment, details in top_keywords.items():
                with st.expander(sentiment):
                    st.write("Positive signal terms:")
                    st.write(", ".join([x["term"] for x in details.get("top_positive_terms", [])[:10]]))
                    st.write("Negative signal terms:")
                    st.write(", ".join([x["term"] for x in details.get("top_negative_terms", [])[:10]]))

    with tabs[3]:
        st.subheader("Dataset Preview")
        st.write("A quick look at the processed dataset used for training and evaluation.")
        st.dataframe(df.head(25), use_container_width=True)

        preview_left, preview_right = st.columns(2)
        with preview_left:
            sentiment_counts = df["label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            pie = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment split")
            st.plotly_chart(pie, use_container_width=True)
        with preview_right:
            source_counts = df["source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            bar = px.bar(source_counts, x="Source", y="Count", title="Source distribution")
            st.plotly_chart(bar, use_container_width=True)

        st.download_button(
            "Download processed dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_dataset_processed.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    app()
