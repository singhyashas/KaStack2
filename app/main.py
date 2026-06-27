import base64
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet

try:
    from supabase import create_client
except Exception:  # pragma: no cover - optional dependency
    create_client = None

try:
    import boto3
except Exception:  # pragma: no cover - optional dependency
    boto3 = None


class Message(BaseModel):
    role: str = Field(..., description="speaker role")
    content: str = Field(..., description="message content")
    created_at: Optional[str] = None
    mood: Optional[str] = None


class UploadRequest(BaseModel):
    user_id: str = Field(..., description="user identifier")
    password: str = Field(..., description="user credential")
    conversation: List[Message] = Field(default_factory=list)


class AskRequest(BaseModel):
    user_id: str = Field(..., description="user identifier")
    password: str = Field(..., description="user credential")
    question: str = Field(..., description="question to answer")
    top_k: int = Field(3, ge=1, le=10)


class InMemoryStore:
    def __init__(self) -> None:
        self.conversations: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.backups: List[Dict[str, Any]] = []


store = InMemoryStore()
app = FastAPI(title="Conversation Memory API", version="1.0.0")


def derive_key(user_id: str, password: str) -> bytes:
    material = f"{user_id}:{password}".encode("utf-8")
    return hashlib.sha256(material).digest()


def encrypt_backup(payload: Dict[str, Any], user_id: str, password: str) -> Dict[str, Any]:
    key = base64.urlsafe_b64encode(derive_key(user_id, password))
    fernet = Fernet(key)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    token = fernet.encrypt(encoded).decode("utf-8")
    decrypted = fernet.decrypt(token.encode("utf-8")).decode("utf-8")
    return {
        "token": token,
        "round_trip_ok": json.loads(decrypted) == payload,
        "decrypted_preview": json.loads(decrypted)["summary"] if "summary" in json.loads(decrypted) else json.loads(decrypted),
    }


def infer_topic(message: str) -> str:
    lowered = message.lower()
    hints = {
        "work": ["work", "project", "deadline", "meeting", "office"],
        "health": ["doctor", "sleep", "pain", "medicine", "exercise"],
        "family": ["family", "mom", "dad", "sister", "brother"],
        "travel": ["trip", "travel", "hotel", "airport", "flight"],
        "finance": ["money", "budget", "invoice", "bank", "invest"],
        "food": ["food", "restaurant", "dinner", "lunch", "coffee"],
        "hobby": ["game", "music", "movie", "book", "hobby"],
    }
    for label, keywords in hints.items():
        if any(word in lowered for word in keywords):
            return label
    return "general"


def infer_mood(message: str) -> Dict[str, Any]:
    lowered = message.lower()
    positive_words = ["happy", "great", "love", "excited", "good", "amazing", "joy"]
    negative_words = ["sad", "angry", "upset", "bad", "tired", "stress", "worried", "pain"]
    score = 0.0
    for word in positive_words:
        if word in lowered:
            score += 0.25
    for word in negative_words:
        if word in lowered:
            score -= 0.25
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": round(score, 2)}


def build_topic_segments(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not messages:
        return []
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_topic: Optional[str] = None
    for message in messages:
        topic = infer_topic(message["content"])
        if current_topic is None:
            current_topic = topic
            current = [message]
            continue
        if topic != current_topic and len(current) >= 5:
            segments.append(current)
            current = [message]
            current_topic = topic
        else:
            current.append(message)
    if current:
        segments.append(current)
    return segments or [messages]


def summarize_segment(messages: List[Dict[str, Any]], segment_type: str) -> str:
    preview = " ".join(item["content"] for item in messages[:3])
    preview = re.sub(r"\s+", " ", preview)[:180]
    if segment_type == "topic":
        return f"Segment with {len(messages)} messages. Preview: {preview}"
    if segment_type == "day":
        return f"Daily summary for {len(messages)} messages. Preview: {preview}"
    return f"Checkpoint with {len(messages)} messages. Preview: {preview}"


def extract_events(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for idx, message in enumerate(messages):
        content = message["content"].lower()
        if "grocery" in content or "shopping list" in content:
            events.append({"id": str(uuid4()), "type": "grocery_list", "message_index": idx, "content": message["content"]})
        if "mood drop" in content or "feeling down" in content:
            events.append({"id": str(uuid4()), "type": "mood_drop", "message_index": idx, "content": message["content"]})
    return events


def build_checkpoint_payload(conversation: List[Dict[str, Any]], user_id: str, password: str) -> Dict[str, Any]:
    segments = build_topic_segments(conversation)
    topic_checkpoints = []
    mood_checkpoints = []
    for index, segment in enumerate(segments, start=1):
        topic_checkpoints.append({
            "id": str(uuid4()),
            "type": "topic",
            "segment": index,
            "summary": summarize_segment(segment, "topic"),
            "message_count": len(segment),
        })
        mood = infer_mood(" ".join(item["content"] for item in segment))
        mood_checkpoints.append({
            "id": str(uuid4()),
            "type": "mood",
            "segment": index,
            "label": mood["label"],
            "score": mood["score"],
            "message_count": len(segment),
        })
    chunked = [conversation[i:i + 100] for i in range(0, len(conversation), 100)] if conversation else []
    hundred_message_checkpoints = [
        {
            "id": str(uuid4()),
            "type": "message_count",
            "segment": index,
            "summary": summarize_segment(chunk, "message_count"),
            "message_count": len(chunk),
        }
        for index, chunk in enumerate(chunked, start=1)
    ]
    day_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in conversation:
        day = (item.get("created_at") or datetime.now(timezone.utc).isoformat()).split("T", 1)[0]
        day_groups.setdefault(day, []).append(item)
    day_checkpoints = [
        {
            "id": str(uuid4()),
            "type": "day",
            "day": day,
            "summary": summarize_segment(items, "day"),
            "message_count": len(items),
        }
        for day, items in sorted(day_groups.items())
    ]
    backup_payload = {
        "user_id": user_id,
        "summary": f"Backup for {len(conversation)} messages",
        "day_checkpoints": day_checkpoints,
        "topic_checkpoints": topic_checkpoints,
        "mood_checkpoints": mood_checkpoints,
    }
    backup_result = encrypt_backup(backup_payload, user_id, password)
    return {
        "topic_checkpoints": topic_checkpoints,
        "hundred_message_checkpoints": hundred_message_checkpoints,
        "day_checkpoints": day_checkpoints,
        "mood_checkpoints": mood_checkpoints,
        "events": extract_events(conversation),
        "backup": backup_result,
    }


def sync_to_supabase(payload: Dict[str, Any], user_id: str) -> None:
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_ANON_KEY") or create_client is None:
        return
    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
    try:
        client.table("conversation_checkpoints").insert({
            "user_id": user_id,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception:
        return


def maybe_upload_to_r2(payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    if not os.getenv("R2_BUCKET") or boto3 is None:
        return {"backend": "memory", "bucket": None}
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        key = f"backups/{user_id}/{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
        client.put_object(Bucket=os.getenv("R2_BUCKET"), Key=key, Body=json.dumps(payload).encode("utf-8"))
        return {"backend": "r2", "bucket": os.getenv("R2_BUCKET"), "key": key}
    except Exception as exc:
        return {"backend": "memory", "bucket": None, "error": str(exc)}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def keyword_matches(question: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokens = set(normalize(question).split())
    if not tokens:
        return documents[:3]
    scored = []
    for doc in documents:
        doc_tokens = set(normalize(doc["text"]).split())
        score = len(tokens & doc_tokens)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:3]]


def semantic_matches(question: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not documents:
        return []
    question_tokens = normalize(question).split()
    scored = []
    for doc in documents:
        doc_tokens = normalize(doc["text"]).split()
        if not question_tokens or not doc_tokens:
            continue
        overlap = Counter(question_tokens) & Counter(doc_tokens)
        score = sum(overlap.values()) / max(len(question_tokens), 1)
        if score > 0.0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:3]]


def route_query(question: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    keyword_hits = keyword_matches(question, documents)
    semantic_hits = semantic_matches(question, documents)
    if keyword_hits and semantic_hits:
        merged = []
        seen = set()
        for doc in keyword_hits + semantic_hits:
            marker = (doc.get("text"), doc.get("source"), doc.get("created_at"))
            if marker not in seen:
                seen.add(marker)
                merged.append(doc)
        return {
            "path": "both",
            "reason": "The question has clear lexical overlap and semantic similarity.",
            "documents": merged,
        }
    if semantic_hits:
        return {
            "path": "semantic",
            "reason": "The question is better answered by semantic similarity than exact keywords.",
            "documents": semantic_hits,
        }
    return {
        "path": "keyword",
        "reason": "The question appears to target explicit terms found in stored messages.",
        "documents": keyword_hits,
    }


def generate_answer(question: str, documents: List[Dict[str, Any]]) -> str:
    if not documents:
        return "I do not have enough stored context to answer that confidently yet."
    snippets = []
    for doc in documents[:3]:
        snippets.append(doc["text"])
    return f"Based on the stored memory, the best match is: {' | '.join(snippets)}"


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/upload")
def upload(payload: UploadRequest) -> Dict[str, Any]:
    if not payload.conversation:
        raise HTTPException(status_code=400, detail="conversation cannot be empty")
    conversation_payload = [
        {
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at or datetime.now(timezone.utc).isoformat(),
            "mood": message.mood,
        }
        for message in payload.conversation
    ]
    checkpoint_payload = build_checkpoint_payload(conversation_payload, payload.user_id, payload.password)
    store.conversations.append({
        "user_id": payload.user_id,
        "messages": conversation_payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    store.checkpoints.append({
        "user_id": payload.user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": checkpoint_payload,
    })
    store.events.extend(checkpoint_payload["events"])
    store.backups.append({"user_id": payload.user_id, "backup": checkpoint_payload["backup"]})
    sync_to_supabase(checkpoint_payload, payload.user_id)
    maybe_upload_to_r2(checkpoint_payload, payload.user_id)
    return {
        "status": "ok",
        "user_id": payload.user_id,
        "message_count": len(conversation_payload),
        "checkpoint_counts": {
            "topic": len(checkpoint_payload["topic_checkpoints"]),
            "hundred_message": len(checkpoint_payload["hundred_message_checkpoints"]),
            "day": len(checkpoint_payload["day_checkpoints"]),
            "mood": len(checkpoint_payload["mood_checkpoints"]),
            "events": len(checkpoint_payload["events"]),
        },
        "backup_round_trip": checkpoint_payload["backup"]["round_trip_ok"],
        "backup_storage": maybe_upload_to_r2(checkpoint_payload, payload.user_id),
    }


@app.post("/ask")
def ask(payload: AskRequest) -> Dict[str, Any]:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    user_messages = [
        item["messages"]
        for item in store.conversations
        if item["user_id"] == payload.user_id
    ]
    flattened_messages = [message for messages in user_messages for message in messages]
    documents = [
        {"text": message["content"], "source": message["role"], "created_at": message.get("created_at")}
        for message in flattened_messages
    ]
    if not documents:
        return {
            "answer": "No stored conversation was found for this user yet.",
            "retrieval_path": "keyword",
            "reason": "No conversation history is available for retrieval.",
            "sources": [],
        }
    routed = route_query(payload.question, documents)
    selected = routed["documents"][: payload.top_k]
    answer = generate_answer(payload.question, selected)
    return {
        "answer": answer,
        "retrieval_path": routed["path"],
        "reason": routed["reason"],
        "sources": [
            {
                "text": doc["text"],
                "source": doc["source"],
                "created_at": doc.get("created_at"),
            }
            for doc in selected
        ],
    }
