from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

RAW_DATA_PATH = RAW_DIR / "sentiment_dataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "sentiment_dataset_processed.csv"
TOP_KEYWORDS_PATH = PROCESSED_DIR / "top_keywords.json"
INSIGHTS_PATH = PROCESSED_DIR / "business_insights.json"

MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
N_SAMPLES = 1500
SENTIMENTS = ["Positive", "Negative", "Neutral"]

BRANDS = [
    "Acme", "Nova", "Pulse", "Zenith", "Orbit", "Vertex", "Nimbus", "Spark", "Atlas", "Summit"
]

PRODUCTS = [
    "app", "website", "service", "delivery", "support team", "checkout", "mobile app",
    "dashboard", "product quality", "customer care", "pricing", "order flow"
]
