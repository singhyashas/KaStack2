"""
checkpointing.py
Hackathon Task 1 — Checkpoints (store in Supabase)

Builds on the Round 1/2 logic (topic detection, 100-message checkpoints,
day-wise segmentation, mood scoring) and writes each checkpoint type into
its corresponding Supabase table.
"""

import os
import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
CSV_PATH = os.environ.get("CSV_PATH", "conversations.csv")


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set in .env")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    global_index: int
    conv_index: int      # "day" index
    msg_index: int
    speaker: str
    text: str


def load_all_messages(csv_path: str) -> List[Message]:
    df = pd.read_csv(csv_path, header=None)
    all_messages = []
    global_idx = 0
    for conv_idx, row in df.iterrows():
        raw = str(row[0])
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        for msg_idx, line in enumerate(lines):
            match = re.match(r"^(User \d+):\s*(.+)$", line)
            if match:
                all_messages.append(Message(
                    global_index=global_idx,
                    conv_index=conv_idx,
                    msg_index=msg_idx,
                    speaker=match.group(1),
                    text=match.group(2).strip(),
                ))
                global_idx += 1
    print(f"[Checkpointing] Loaded {len(df)} conversations -> {len(all_messages)} messages")
    return all_messages


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lightweight extractive summarization (TF-IDF, no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = set([
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "a",
    "an", "the", "and", "but", "or", "so", "to", "of", "in", "on", "at",
    "for", "with", "about", "like", "from", "by", "as", "its", "just",
])


def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-z]+\b", text.lower())
            if w not in STOPWORDS and len(w) > 2]


def summarize(messages: List[Message], max_sentences: int = 3) -> str:
    if not messages:
        return ""
    sentences = [f"{m.speaker}: {m.text}" for m in messages]
    if len(sentences) <= max_sentences:
        return " | ".join(sentences)

    doc_tokens = [tokenize(m.text) for m in messages]
    N = len(messages)
    df_counts = Counter()
    for tokens in doc_tokens:
        for w in set(tokens):
            df_counts[w] += 1
    idf = {w: math.log((N + 1) / (c + 1)) for w, c in df_counts.items()}

    scores = []
    for tokens in doc_tokens:
        if not tokens:
            scores.append(0.0)
            continue
        tf = Counter(tokens)
        total = sum(tf.values())
        scores.append(sum((tf[w] / total) * idf.get(w, 0) for w in tf))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_idx = sorted(i for i, _ in ranked[:max_sentences])
    return " ".join(sentences[i] for i in top_idx)


def extract_keywords(messages: List[Message], top_n: int = 5) -> List[str]:
    words = []
    for m in messages:
        words.extend(tokenize(m.text))
    return [w for w, _ in Counter(words).most_common(top_n)]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Topic checkpoints — keyword-overlap window detection (fast, no deps)
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 8
MIN_SEGMENT_SIZE = 5


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_topics(messages: List[Message]) -> List[Dict[str, Any]]:
    """Returns list of {topic_id, start_idx, end_idx, messages}."""
    segments = []
    topic_id = 1
    seg_start = 0
    i = WINDOW_SIZE

    while i < len(messages):
        before = set()
        for m in messages[max(0, i - WINDOW_SIZE):i]:
            before.update(tokenize(m.text))
        after = set()
        for m in messages[i:min(len(messages), i + WINDOW_SIZE)]:
            after.update(tokenize(m.text))

        sim = jaccard(before, after)
        segment_len = i - seg_start

        if sim < 0.15 and segment_len >= MIN_SEGMENT_SIZE:
            seg_msgs = messages[seg_start:i]
            segments.append({
                "topic_id": topic_id,
                "start_idx": seg_msgs[0].global_index,
                "end_idx": seg_msgs[-1].global_index,
                "messages": seg_msgs,
            })
            topic_id += 1
            seg_start = i
            i += WINDOW_SIZE
        else:
            i += 1

    if seg_start < len(messages):
        seg_msgs = messages[seg_start:]
        segments.append({
            "topic_id": topic_id,
            "start_idx": seg_msgs[0].global_index,
            "end_idx": seg_msgs[-1].global_index,
            "messages": seg_msgs,
        })
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mood scoring (lexicon-based)
# ─────────────────────────────────────────────────────────────────────────────

MOOD_LEXICON = {
    "happy": [r"\b(happy|excited|thrilled|amazing|great|wonderful|glad|fantastic)\b"],
    "sad": [r"\b(sad|down|upset|crying|heartbroken|lonely|depressed)\b"],
    "anxious": [r"\b(worried|nervous|stressed|overwhelmed|anxious|scared|afraid)\b"],
    "angry": [r"\b(angry|furious|mad|pissed|hate|infuriat)\b"],
    "calm": [r"\b(calm|relaxed|peaceful|fine|okay|alright|content)\b"],
    "playful": [r"\b(lol|haha|lmao|joke|funny|hilarious)\b"],
}

TONE_LEXICON = {
    "formal": [r"\b(would you|could you please|i would like|regarding)\b"],
    "casual": [r"\b(lol|yeah|gonna|wanna|kinda|hey|bro)\b"],
}


def score_mood(messages: List[Message]) -> Dict[str, Any]:
    text = " ".join(m.text for m in messages).lower()
    mood_scores = Counter()
    for label, patterns in MOOD_LEXICON.items():
        for pat in patterns:
            mood_scores[label] += len(re.findall(pat, text))
    tone_scores = Counter()
    for label, patterns in TONE_LEXICON.items():
        for pat in patterns:
            tone_scores[label] += len(re.findall(pat, text))

    total = sum(mood_scores.values())
    if total == 0:
        return {"mood_label": "neutral", "mood_score": 0.0, "tone_label": "neutral"}

    dominant_mood = max(mood_scores, key=mood_scores.get)
    mood_confidence = mood_scores[dominant_mood] / total
    dominant_tone = max(tone_scores, key=tone_scores.get) if sum(tone_scores.values()) > 0 else "neutral"

    return {
        "mood_label": dominant_mood,
        "mood_score": round(mood_confidence, 3),
        "tone_label": dominant_tone,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint builders -> Supabase writers
# ─────────────────────────────────────────────────────────────────────────────

def build_and_store_topic_checkpoints(sb: Client, messages: List[Message], user_id: Optional[str] = None):
    segments = detect_topics(messages)
    rows = []
    for seg in segments:
        rows.append({
            "user_id": user_id,
            "topic_id": seg["topic_id"],
            "start_msg_idx": seg["start_idx"],
            "end_msg_idx": seg["end_idx"],
            "summary": summarize(seg["messages"]),
            "keywords": extract_keywords(seg["messages"]),
        })
    if rows:
        sb.table("topic_checkpoints").insert(rows).execute()
    print(f"[Checkpointing] Stored {len(rows)} topic checkpoints")
    return segments


def build_and_store_message_checkpoints(sb: Client, messages: List[Message],
                                          interval: int = 100, user_id: Optional[str] = None):
    rows = []
    ckpt_id = 1
    for start in range(0, len(messages), interval):
        chunk = messages[start:start + interval]
        rows.append({
            "user_id": user_id,
            "checkpoint_id": ckpt_id,
            "start_msg_idx": chunk[0].global_index,
            "end_msg_idx": chunk[-1].global_index,
            "message_count": len(chunk),
            "summary": summarize(chunk, max_sentences=5),
        })
        ckpt_id += 1
    if rows:
        sb.table("message_checkpoints").insert(rows).execute()
    print(f"[Checkpointing] Stored {len(rows)} 100-message checkpoints")


def build_and_store_mood_checkpoints(sb: Client, messages: List[Message], user_id: Optional[str] = None):
    """One mood checkpoint per day (conv_index)."""
    days: Dict[int, List[Message]] = {}
    for m in messages:
        days.setdefault(m.conv_index, []).append(m)

    rows = []
    for day_idx, day_msgs in days.items():
        mood = score_mood(day_msgs)
        rows.append({
            "user_id": user_id,
            "day_index": day_idx,
            "segment_start_idx": day_msgs[0].global_index,
            "segment_end_idx": day_msgs[-1].global_index,
            "mood_label": mood["mood_label"],
            "mood_score": mood["mood_score"],
            "tone_label": mood["tone_label"],
        })
    if rows:
        for i in range(0, len(rows), 500):
            sb.table("mood_checkpoints").insert(rows[i:i + 500]).execute()
    print(f"[Checkpointing] Stored {len(rows)} mood checkpoints")


def build_day_checkpoint_metadata(messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Returns day-level metadata. encryption.py handles encrypting + uploading
    to R2 first, then writes the row with the resulting r2_backup_key +
    encryption_iv into day_checkpoints.
    """
    days: Dict[int, List[Message]] = {}
    for m in messages:
        days.setdefault(m.conv_index, []).append(m)

    day_meta = []
    for day_idx, day_msgs in days.items():
        day_meta.append({
            "day_index": day_idx,
            "message_count": len(day_msgs),
            "summary": summarize(day_msgs, max_sentences=3),
            "messages": day_msgs,
        })
    return day_meta


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all_checkpointing(csv_path: str = CSV_PATH, max_messages: Optional[int] = None,
                           user_id: Optional[str] = None):
    sb = get_supabase_client()
    messages = load_all_messages(csv_path)
    if max_messages:
        messages = messages[:max_messages]

    print("[Checkpointing] Building topic checkpoints...")
    build_and_store_topic_checkpoints(sb, messages, user_id=user_id)

    print("[Checkpointing] Building 100-message checkpoints...")
    build_and_store_message_checkpoints(sb, messages, user_id=user_id)

    print("[Checkpointing] Building mood checkpoints...")
    build_and_store_mood_checkpoints(sb, messages, user_id=user_id)

    print("[Checkpointing] Building day-checkpoint metadata (for encryption module)...")
    day_meta = build_day_checkpoint_metadata(messages)
    print(f"[Checkpointing] {len(day_meta)} days ready for encryption + R2 upload")

    return {"messages": messages, "day_metadata": day_meta}


if __name__ == "__main__":
    result = run_all_checkpointing(max_messages=2000)
    print("\nDone. Day metadata sample:")
    print({k: v for k, v in result["day_metadata"][0].items() if k != "messages"})
