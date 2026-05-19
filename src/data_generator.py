from __future__ import annotations

from datetime import datetime, timedelta
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, N_SAMPLES, RANDOM_STATE, BRANDS, PRODUCTS

POSITIVE_OPENERS = [
    "absolutely love",
    "really like",
    "am genuinely happy with",
    "had a great experience with",
    "was pleasantly surprised by",
    "feel good about",
    "am impressed by",
]
NEGATIVE_OPENERS = [
    "really disappointed with",
    "am frustrated by",
    "had a bad experience with",
    "keep running into issues with",
    "feel stuck with",
    "was honestly a mess",
    "cannot trust",
]
NEUTRAL_OPENERS = [
    "checked",
    "tried",
    "used",
    "looked at",
    "reviewed",
    "tested",
    "went through",
]

POSITIVE_ASPECTS = [
    "pricing", "support", "checkout", "delivery", "app", "dashboard", "onboarding", "response time", "design", "quality"
]
NEGATIVE_ASPECTS = [
    "pricing", "support", "checkout", "delivery", "app", "dashboard", "login", "payment", "refunds", "notifications"
]
NEUTRAL_ASPECTS = [
    "pricing", "support", "dashboard", "setup", "product page", "mobile app", "order flow", "site layout", "feature list", "help center"
]

POSITIVE_ENDINGS = [
    "It saved me a lot of time.",
    "The overall flow feels smooth.",
    "The experience felt worth it.",
    "I would use it again.",
    "I had no real complaints.",
]
NEGATIVE_ENDINGS = [
    "I still had to repeat myself twice.",
    "The whole process felt annoying.",
    "It slowed everything down.",
    "I would not recommend it yet.",
    "It honestly felt like a waste of time.",
]
NEUTRAL_ENDINGS = [
    "Nothing really stood out.",
    "It felt pretty average overall.",
    "I did not notice anything major.",
    "It seems acceptable for now.",
    "That is about it.",
]

POSITIVE_ADJ = ["fast", "clean", "helpful", "simple", "easy", "reliable", "smooth", "useful", "nice", "solid"]
NEGATIVE_ADJ = ["slow", "buggy", "confusing", "messy", "frustrating", "glitchy", "unhelpful", "broken", "awkward", "clunky"]
NEUTRAL_ADJ = ["okay", "average", "fine", "standard", "basic", "ordinary", "decent", "acceptable", "middle of the road", "simple"]

POSITIVE_HASHTAGS = ["#loveit", "#goodvibes", "#happy", "#worthit", "#smooth"]
NEGATIVE_HASHTAGS = ["#fail", "#frustrated", "#buggy", "#annoyed", "#needsfixing"]
NEUTRAL_HASHTAGS = ["#update", "#review", "#feedback", "#test", "#status"]

SOURCES = ["Twitter", "Review", "Comment"]


def _choose_date(idx: int, total: int, rng: random.Random) -> str:
    start = datetime.now() - timedelta(days=180)
    delta_days = int(180 * idx / max(total - 1, 1))
    jitter = rng.randint(0, 4)
    dt = start + timedelta(days=delta_days + jitter)
    return dt.strftime("%Y-%m-%d")


def _maybe_add_social_noise(text: str, label: str, rng: random.Random) -> str:
    if rng.random() < 0.20:
        text += rng.choice(["!", "!!", "...", " :)", " 😅", " 🙃"])
    if rng.random() < 0.12:
        text += " " + rng.choice(["honestly", "fr", "tbh", "just saying", "for real", "ngl"])
    if rng.random() < 0.12:
        text += " " + rng.choice(["pls respond", "any update?", "someone help", "fix this", "great job", "thanks team"])
    if label == "Positive" and rng.random() < 0.22:
        text += " " + rng.choice(POSITIVE_HASHTAGS)
    elif label == "Negative" and rng.random() < 0.22:
        text += " " + rng.choice(NEGATIVE_HASHTAGS)
    elif label == "Neutral" and rng.random() < 0.16:
        text += " " + rng.choice(NEUTRAL_HASHTAGS)
    return text


def _build_positive_text(brand: str, product: str, rng: random.Random) -> str:
    opener = rng.choice(POSITIVE_OPENERS)
    aspect = rng.choice(POSITIVE_ASPECTS)
    adjective = rng.choice(POSITIVE_ADJ)
    ending = rng.choice(POSITIVE_ENDINGS)
    templates = [
        f"I {opener} {brand}'s {product}; the {aspect} feels {adjective}. {ending}",
        f"{brand}'s {product} is {adjective} when it comes to {aspect}. {ending}",
        f"Honestly, the {aspect} on {brand} is {adjective}. {ending}",
        f"The latest update from {brand} feels {adjective}. The {aspect} part is {adjective}. {ending}",
        f"{product.capitalize()} from {brand} was {adjective}, especially the {aspect}. {ending}",
        f"{brand}'s {product} keeps getting better. The {aspect} is {adjective}. {ending}",
    ]
    text = rng.choice(templates)
    if rng.random() < 0.25:
        text += " " + rng.choice([
            "Small issues, but nothing that ruined it.",
            "A couple of rough edges, still positive overall.",
            "Not perfect, but definitely better than expected.",
        ])
    return _maybe_add_social_noise(text, "Positive", rng)


def _build_negative_text(brand: str, product: str, rng: random.Random) -> str:
    opener = rng.choice(NEGATIVE_OPENERS)
    aspect = rng.choice(NEGATIVE_ASPECTS)
    adjective = rng.choice(NEGATIVE_ADJ)
    ending = rng.choice(NEGATIVE_ENDINGS)
    templates = [
        f"I {opener} {brand}'s {product}; the {aspect} feels {adjective}. {ending}",
        f"{brand}'s {product} feels {adjective} around {aspect}. {ending}",
        f"The {aspect} on {brand} is {adjective}. {ending}",
        f"This version of the {product} from {brand} is {adjective}; the {aspect} is still a problem. {ending}",
        f"{product.capitalize()} from {brand} was {adjective}, especially the {aspect}. {ending}",
        f"{brand}'s {product} keeps causing problems. The {aspect} is {adjective}. {ending}",
    ]
    text = rng.choice(templates)
    if rng.random() < 0.30:
        text += " " + rng.choice([
            "Support kept sending the same reply.",
            "I had to refresh the page a few times.",
            "This needs a proper fix before launch.",
            "The whole thing felt more stressful than it should have been.",
        ])
    return _maybe_add_social_noise(text, "Negative", rng)


def _build_neutral_text(brand: str, product: str, rng: random.Random) -> str:
    opener = rng.choice(NEUTRAL_OPENERS)
    aspect = rng.choice(NEUTRAL_ASPECTS)
    adjective = rng.choice(NEUTRAL_ADJ)
    ending = rng.choice(NEUTRAL_ENDINGS)
    templates = [
        f"I {opener} {brand}'s {product} today; the {aspect} looks {adjective}. {ending}",
        f"{brand}'s {product} is {adjective} for the {aspect}. {ending}",
        f"The {aspect} on {brand} seems {adjective}. {ending}",
        f"I used the {product} from {brand} for a bit and it felt {adjective}. {ending}",
        f"{product.capitalize()} from {brand} is {adjective} overall. {ending}",
    ]
    text = rng.choice(templates)
    if rng.random() < 0.25:
        text += " " + rng.choice([
            "Nothing major to flag.",
            "It looks fine for everyday use.",
            "I might need more time before judging it properly.",
            "There are a few positives and a few limits.",
        ])
    return _maybe_add_social_noise(text, "Neutral", rng)


def generate_synthetic_dataset(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    rows = []
    class_counts = {
        "Positive": n_samples // 3,
        "Negative": n_samples // 3,
        "Neutral": n_samples - 2 * (n_samples // 3),
    }

    for label, count in class_counts.items():
        for _ in range(count):
            brand = rng.choice(BRANDS)
            product = rng.choice(PRODUCTS)
            if label == "Positive":
                text = _build_positive_text(brand, product, rng)
            elif label == "Negative":
                text = _build_negative_text(brand, product, rng)
            else:
                text = _build_neutral_text(brand, product, rng)

            if rng.random() < 0.12:
                text = text.replace(".", "!")
            if rng.random() < 0.08:
                text += f" ({rng.choice(['tweet', 'review', 'comment'])})"

            rows.append({
                "id": len(rows) + 1,
                "date": _choose_date(len(rows), n_samples, rng),
                "source": rng.choice(SOURCES),
                "brand": brand,
                "product": product,
                "text": text,
                "label": label,
            })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Add a small amount of extra natural variation without shrinking the dataset.
    extra_noise_idx = np_rng.choice(df.index, size=max(30, n_samples // 15), replace=False)
    noise_tail = ["", "!!", "??", " right now", " asap", " honestly", " tbh"]
    for idx in extra_noise_idx:
        df.loc[idx, "text"] = f"{df.loc[idx, 'text']}{rng.choice(noise_tail)}"

    # Keep the task realistic with a little label noise.
    flip_count = max(30, int(n_samples * 0.07))
    flip_idx = np_rng.choice(df.index, size=flip_count, replace=False)
    flip_map = {"Positive": "Negative", "Negative": "Neutral", "Neutral": "Positive"}
    for idx in flip_idx:
        df.loc[idx, "label"] = flip_map[df.loc[idx, "label"]]

    return df


def save_dataset(path: Path = RAW_DATA_PATH, n_samples: int = N_SAMPLES) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_dataset(n_samples=n_samples)
    df.to_csv(path, index=False)
    return df
