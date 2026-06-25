from collections import Counter, defaultdict

from persona_utils import infer_baseline_tones


MOOD_KEYWORDS = {
    "curious": {"why", "how", "learn", "understand", "explore", "curious", "wondering"},
    "formal": {"please", "could", "would", "schedule", "review", "prepare", "professor"},
    "casual": {"hey", "yeah", "okay", "cool", "gonna", "kinda", "btw"},
    "frustrated": {"stuck", "annoyed", "failing", "failed", "confused", "frustrating", "tired"},
    "playful": {"haha", "fun", "joke", "playful", "lol", "nice"},
    "anxious": {"worried", "nervous", "anxious", "deadline", "pressure", "scared"},
    "relieved": {"better", "relieved", "calm", "sorted", "fixed", "fine"},
}

TRIGGER_KEYWORDS = {
    "project": {"project", "bug", "demo", "build", "submission", "failing"},
    "exam": {"exam", "placement", "interview", "test"},
    "deadline": {"deadline", "tomorrow", "tonight", "late"},
    "sister": {"sister"},
    "professor": {"professor"},
    "friend": {"friend"},
}

TOPIC_KEYWORDS = {
    "learning": {"memory", "systems", "context", "understand", "learn"},
    "college": {"placement", "exam", "professor", "checklist"},
    "family": {"sister", "family", "argument", "called", "miss"},
    "project": {"project", "retrieval", "resolver", "demo", "build"},
    "deadline": {"deadline", "tomorrow", "tonight", "submit"},
    "action": {"send", "prepare", "create", "upload", "finish"},
    "small-talk": {"haha", "cool", "hey", "fun", "okay"},
}


def _tokenize(text):
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def _score_day(messages):
    tokens = []
    for message in messages:
        tokens.extend(_tokenize(message["message"]))

    token_set = set(tokens)
    scores = {
        mood: len(words.intersection(token_set))
        for mood, words in MOOD_KEYWORDS.items()
    }

    top = [mood for mood, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score > 0]
    if not top:
        top = ["neutral"]

    return top[:2], scores


def _detect_trigger(messages):
    day_text = " ".join(message["message"].lower() for message in messages)
    matches = []
    for trigger, keywords in TRIGGER_KEYWORDS.items():
        if any(keyword in day_text for keyword in keywords):
            matches.append(trigger)

    if matches:
        return ", ".join(matches[:2])
    return "tone shift in conversation"


def build_persona_timeline(messages, persona=None, speaker=None):
    grouped = defaultdict(list)
    for message in messages:
        if message["speaker"] == "user":
            grouped[message["date"]].append(message)

    timeline = []
    previous_state = None
    baseline = " & ".join(infer_baseline_tones(persona, speaker)) if persona else "-"

    for index, date in enumerate(sorted(grouped.keys()), start=1):
        moods, _scores = _score_day(grouped[date])
        state = " & ".join(moods)
        drift = previous_state is not None and state != previous_state

        timeline.append(
            {
                "day": f"Day {index}",
                "date": date,
                "persona_state": state,
                "round1_baseline": baseline,
                "drift_detected": drift,
                "trigger": _detect_trigger(grouped[date]) if drift else "-",
                "message_count": len(grouped[date]),
            }
        )
        previous_state = state

    return timeline


def summarize_topics(messages):
    topics = Counter()
    for message in messages:
        topic = split_topic(message["message"])
        topics[topic] += 1
    return topics.most_common()


def split_topic(text):
    tokens = set(_tokenize(text))
    scores = {
        topic: len(tokens.intersection(keywords))
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best_topic, best_score = max(scores.items(), key=lambda item: item[1])
    return best_topic if best_score > 0 else "general"


def topic_split_table(messages):
    rows = []
    for index, message in enumerate(messages, start=1):
        if message["speaker"] != "user":
            continue
        detected_topic = split_topic(message["message"])
        rows.append(
            {
                "message_id": index,
                "date": message["date"],
                "stored_topic": message.get("topic", "general"),
                "detected_topic": detected_topic,
                "message": message["message"],
            }
        )
    return rows
