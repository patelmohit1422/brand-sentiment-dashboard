from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_model import train_and_save_model
from src.visualize import create_visualizations
from src.insights import build_insights


def main():
    df, metrics, top_keywords = train_and_save_model(force_regenerate_data=True)
    create_visualizations(df, top_keywords)
    build_insights(df, metrics, top_keywords)
    print("Project built successfully.")
    print(f"Samples: {len(df):,}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
