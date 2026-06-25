from datetime import datetime


EMOTION_WEIGHTS = {
    "frustrated": 0.9,
    "anxious": 0.85,
    "sad": 0.8,
    "relieved": 0.72,
    "curious": 0.55,
    "playful": 0.45,
    "neutral": 0.35,
}

NEGATIVE_TERMS = {"miss", "not talking", "argued", "worried", "upset", "angry", "hurt"}
POSITIVE_TERMS = {"called", "better", "laughed", "fine", "relieved", "sorted", "happy"}


def _extract_query_terms(question):
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in question)
    tokens = [token for token in cleaned.split() if len(token) > 2]
    relation_terms = {"sister", "brother", "mother", "father", "friend", "professor"}
    matched = [token for token in tokens if token in relation_terms]
    return matched or tokens


def _message_emotion(text):
    lower = text.lower()
    if any(term in lower for term in ["frustrating", "annoyed", "not talking", "argued"]):
        return "frustrated"
    if any(term in lower for term in ["worried", "nervous", "deadline"]):
        return "anxious"
    if any(term in lower for term in ["miss", "tired"]):
        return "sad"
    if any(term in lower for term in ["better", "relieved", "fixed"]):
        return "relieved"
    if any(term in lower for term in ["haha", "fun", "lol"]):
        return "playful"
    if any(term in lower for term in ["how", "why", "learn"]):
        return "curious"
    return "neutral"


def _recency_scores(chunks):
    dates = [datetime.fromisoformat(chunk["date"]) for chunk in chunks]
    earliest = min(dates)
    latest = max(dates)
    span = max((latest - earliest).days, 1)

    scores = {}
    for chunk in chunks:
        current = datetime.fromisoformat(chunk["date"])
        scores[chunk["id"]] = ((current - earliest).days / span)
    return scores


def _find_contradictions(chunks):
    has_negative = any(any(term in chunk["text"].lower() for term in NEGATIVE_TERMS) for chunk in chunks)
    has_positive = any(any(term in chunk["text"].lower() for term in POSITIVE_TERMS) for chunk in chunks)

    if has_negative and has_positive:
        return [
            "Older memories contain tension or distance, while newer memories suggest contact or improvement."
        ]
    return []


def _build_answer(question, ranked_chunks, contradictions):
    if not ranked_chunks:
        return "I could not find a relevant memory for that question."

    subject = ", ".join(_extract_query_terms(question))
    ordered = sorted(ranked_chunks, key=lambda item: item["date"])
    latest = ranked_chunks[0]

    details = " ".join(
        f"On {chunk['date']}, you said: {chunk['text']}"
        for chunk in ordered
    )

    if contradictions:
        return (
            f"Yes. You mentioned {subject} across multiple memories. {details} "
            f"The latest memory is from {latest['date']}, so I would treat that as the current state while still noting the earlier tension."
        )

    return f"Yes. You mentioned {subject}. {details}"


def resolve_memory_question(question, messages):
    terms = _extract_query_terms(question)
    chunks = []

    for index, message in enumerate(messages):
        if message["speaker"] != "user":
            continue

        text = message["message"]
        lower = text.lower()
        keyword_match = sum(1 for term in terms if term in lower)
        if keyword_match == 0:
            continue

        emotion = _message_emotion(text)
        chunks.append(
            {
                "id": index,
                "date": message["date"],
                "topic": message.get("topic", "general"),
                "text": text,
                "emotion": emotion,
                "emotion_weight": EMOTION_WEIGHTS[emotion],
                "keyword_match": keyword_match,
            }
        )

    if not chunks:
        return {
            "answer": "I could not find anything relevant.",
            "ranked_chunks": [],
            "contradictions": [],
        }

    recency = _recency_scores(chunks)
    ranked = []
    for chunk in chunks:
        score = (
            0.55 * recency[chunk["id"]]
            + 0.35 * chunk["emotion_weight"]
            + 0.10 * min(chunk["keyword_match"], 1)
        )
        ranked.append({**chunk, "recency_score": round(recency[chunk["id"]], 3), "final_score": round(score, 3)})

    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    contradictions = _find_contradictions(ranked)

    return {
        "answer": _build_answer(question, ranked, contradictions),
        "ranked_chunks": ranked,
        "contradictions": contradictions,
    }
