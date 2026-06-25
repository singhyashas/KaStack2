import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "intent_training_data.csv"
MODEL_PATH = ROOT / "models" / "intent_model.json"


def _tokenize(text):
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def train_if_needed():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        return

    rows = list(csv.DictReader(DATA_PATH.open("r", encoding="utf-8")))
    class_counts = Counter()
    token_counts = defaultdict(Counter)
    total_tokens = Counter()
    vocabulary = set()

    for row in rows:
        label = row["intent"]
        class_counts[label] += 1
        tokens = _tokenize(row["message"])
        token_counts[label].update(tokens)
        total_tokens[label] += len(tokens)
        vocabulary.update(tokens)

    model = {
        "class_counts": dict(class_counts),
        "token_counts": {label: dict(counts) for label, counts in token_counts.items()},
        "total_tokens": dict(total_tokens),
        "vocabulary": sorted(vocabulary),
        "training_rows": len(rows),
    }
    MODEL_PATH.write_text(json.dumps(model), encoding="utf-8")


def _load_model():
    train_if_needed()
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _predict_scores(model, message):
    tokens = _tokenize(message)
    vocabulary_size = max(len(model["vocabulary"]), 1)
    total_docs = sum(model["class_counts"].values())
    scores = {}

    for label, count in model["class_counts"].items():
        log_prob = math.log(count / total_docs)
        label_token_counts = model["token_counts"][label]
        label_total = model["total_tokens"][label]

        for token in tokens:
            token_count = label_token_counts.get(token, 0)
            log_prob += math.log((token_count + 1) / (label_total + vocabulary_size))

        scores[label] = log_prob

    return scores


def _confidence(scores):
    max_score = max(scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exp_scores.values())
    return {label: value / total for label, value in exp_scores.items()}


def classify_intent(message):
    model = _load_model()

    start = time.perf_counter()
    scores = _predict_scores(model, message)
    probabilities = _confidence(scores)
    intent = max(probabilities, key=probabilities.get)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "message": message,
        "intent": intent,
        "confidence": probabilities[intent],
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
    return results


def benchmark_summary(messages=None):
    frame = benchmark(messages)
    latencies = [row["latency_ms"] for row in frame]
    return {
        "sample_count": len(frame),
        "average_latency_ms": round(sum(latencies) / len(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "model_size_mb": get_model_stats()["model_size_mb"],
    }
