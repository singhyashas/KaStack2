import csv
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "intent_training_data.csv"
MODEL_PATH = ROOT / "models" / "intent_model.joblib"


def train_if_needed():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        return

    rows = list(csv.DictReader(DATA_PATH.open("r", encoding="utf-8")))
    texts = [row["message"] for row in rows]
    labels = [row["intent"] for row in rows]

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")),
        ]
    )
    model.fit(texts, labels)
    joblib.dump(model, MODEL_PATH)


def classify_intent(message):
    train_if_needed()
    model = joblib.load(MODEL_PATH)

    start = time.perf_counter()
    intent = model.predict([message])[0]

    confidence = 1.0
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        confidence = float(model.predict_proba([message]).max())

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "message": message,
        "intent": intent,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }


def get_model_stats():
    train_if_needed()
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    return {
        "model_size_mb": round(size_mb, 3),
        "offline": "yes",
    }


def benchmark(messages=None):
    examples = messages or [
        "Remind me to submit the form tonight",
        "I feel overwhelmed and need support",
        "Send the notes to my team",
        "Hey, how are you doing?",
        "blue rectangle memory cloud",
    ]
    results = [classify_intent(message) for message in examples]
    return pd.DataFrame(results)


def benchmark_summary(messages=None):
    frame = benchmark(messages)
    return {
        "sample_count": len(frame),
        "average_latency_ms": round(float(frame["latency_ms"].mean()), 3),
        "max_latency_ms": round(float(frame["latency_ms"].max()), 3),
        "model_size_mb": get_model_stats()["model_size_mb"],
    }
