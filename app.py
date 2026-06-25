import json
from pathlib import Path

import pandas as pd
import streamlit as st

from intent_classifier import benchmark_summary, classify_intent, get_model_stats, train_if_needed
from persona_engine import build_persona_timeline, topic_split_table
from persona_utils import speaker_names, summarize_round1_persona
from rag_resolver import resolve_memory_question


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_conversations.json"
PERSONA_PATH = ROOT / "data" / "round1_persona.json"


st.set_page_config(
    page_title="Adaptive Persona Memory Engine",
    layout="wide",
)


@st.cache_data
def load_conversations():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_persona():
    with PERSONA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def render_overview(messages, persona, speaker):
    timeline = build_persona_timeline(messages, persona, speaker)
    drift_count = sum(1 for item in timeline if item["drift_detected"])
    model_stats = get_model_stats()
    benchmark = benchmark_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversation Days", len(timeline))
    col2.metric("Drifts Detected", drift_count)
    col3.metric("Model Size", f"{model_stats['model_size_mb']} MB")
    col4.metric("Avg Latency", f"{benchmark['average_latency_ms']} ms")


def render_dataset_tab(messages):
    st.subheader("Conversation Dataset")
    st.caption("A compact multi-day conversation set used to test drift, intent, and memory resolution.")
    topics = pd.DataFrame(messages)["topic"].value_counts().reset_index()
    topics.columns = ["topic", "messages"]

    left, right = st.columns([3, 1])
    with left:
        st.dataframe(pd.DataFrame(messages), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Topic Split")
        st.dataframe(topics, use_container_width=True, hide_index=True)

    st.markdown("#### Detected Topic Checkpoints")
    st.dataframe(pd.DataFrame(topic_split_table(messages)), use_container_width=True, hide_index=True)


def render_persona_json_tab(persona):
    st.subheader("Round 1 Persona JSON")
    speakers = speaker_names(persona)
    selected = st.selectbox("Speaker", speakers)
    summary = summarize_round1_persona(persona, selected)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Messages", summary["message_count"])
    col2.metric("Avg Words", summary["average_content_words"])
    col3.metric("Question Rate", summary["question_rate"])
    col4.metric("Exclamation Rate", summary["exclamation_rate"])

    st.markdown("#### Communication Style")
    st.write(
        {
            "style_notes": summary["style_notes"],
            "top_terms": summary["top_terms"],
            "source": summary["source"],
            "schema_version": summary["schema_version"],
        }
    )

    left, middle, right = st.columns(3)
    with left:
        st.markdown("#### Top Traits")
        st.dataframe(summary["top_traits"], use_container_width=True, hide_index=True)
    with middle:
        st.markdown("#### Top Facts")
        st.dataframe(summary["top_facts"], use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Top Preferences")
        st.dataframe(summary["top_preferences"], use_container_width=True, hide_index=True)

    with st.expander("View raw JSON"):
        st.json(persona)

    return selected


def render_persona_tab(messages, persona, speaker):
    st.subheader("Persona Drift Timeline")
    timeline = build_persona_timeline(messages, persona, speaker)
    timeline_frame = pd.DataFrame(timeline)

    col1, col2, col3 = st.columns(3)
    col1.metric("First State", timeline[0]["persona_state"])
    col2.metric("Latest State", timeline[-1]["persona_state"])
    col3.metric("Drift Events", int(timeline_frame["drift_detected"].sum()))

    st.dataframe(timeline_frame, use_container_width=True, hide_index=True)

    drifts = [item for item in timeline if item["drift_detected"]]
    st.markdown("#### Detected Drifts")
    if not drifts:
        st.info("No major drift detected.")
    else:
        for item in drifts:
            st.write(
                f"**{item['day']} ({item['date']})** changed to **{item['persona_state']}**. "
                f"Likely trigger: **{item['trigger']}**."
            )


def render_intent_tab():
    st.subheader("Offline Intent Classifier")
    train_if_needed()
    stats = get_model_stats()

    left, right = st.columns([2, 1])
    with left:
        examples = [
            "Remind me to call my sister tomorrow evening",
            "I feel overwhelmed and need support",
            "Send the final notes to my team",
            "Hey, how are you doing?",
            "blue rectangle memory cloud",
        ]
        selected = st.selectbox("Try a sample", examples)
        message = st.text_input(
            "Message",
            value=selected,
        )
        if st.button("Classify", type="primary"):
            result = classify_intent(message)
            m1, m2, m3 = st.columns(3)
            m1.metric("Intent", result["intent"])
            m2.metric("Confidence", round(result["confidence"], 3))
            m3.metric("Latency", f"{round(result['latency_ms'], 3)} ms")

    with right:
        st.markdown("#### Model Stats")
        benchmark = benchmark_summary()
        st.write(f"Model size: **{stats['model_size_mb']} MB**")
        st.write(f"Offline: **{stats['offline']}**")
        st.write(f"Average latency: **{benchmark['average_latency_ms']} ms**")
        st.write(f"Max benchmark latency: **{benchmark['max_latency_ms']} ms**")
        st.write(f"Target latency: **< 200 ms/message**")


def render_resolver_tab(messages):
    st.subheader("Conflict-Aware Memory Resolver")
    st.caption("Ranking formula: 0.55 recency + 0.35 emotional weight + 0.10 keyword match.")
    question = st.text_input("Question", value="Did I mention anything about my sister?")

    if st.button("Resolve Memory", type="primary"):
        result = resolve_memory_question(question, messages)
        st.markdown("#### Answer")
        st.success(result["answer"])

        st.markdown("#### Retrieved Chunks")
        columns = [
            "date",
            "topic",
            "text",
            "emotion",
            "emotion_weight",
            "recency_score",
            "final_score",
        ]
        ranked = pd.DataFrame(result["ranked_chunks"])
        if not ranked.empty:
            st.dataframe(ranked[columns], use_container_width=True, hide_index=True)
        else:
            st.info("No relevant chunks found.")

        st.markdown("#### Contradictions")
        if result["contradictions"]:
            for contradiction in result["contradictions"]:
                st.warning(contradiction)
        else:
            st.info("No contradiction detected.")


def render_design_tab():
    st.subheader("System Design Summary")
    st.markdown(
        """
        - Raw conversation history stays on-device.
        - The sync layer stores compact topic checkpoints, timestamps, mood labels, and conflict metadata.
        - Sensitive text can remain local while summaries sync across devices.
        - Memory conflicts are preserved and flagged instead of being overwritten.
        - The resolver ranks evidence using recency, emotional weight, and query relevance.
        """
    )

    st.code(
        """
User Device
  raw messages + local model
        |
        v
Local Memory Engine
  persona drift, intent, retrieval
        |
        v
Sync Server
  summaries + metadata only
        |
        v
Second Device
  rebuilds local memory view
        """,
        language="text",
    )


def main():
    st.title("Adaptive Persona Memory Engine")
    st.caption("Persona drift detection, offline intent classification, and conflict-aware memory retrieval.")

    messages = load_conversations()
    persona = load_persona()
    selected_speaker = speaker_names(persona)[0]
    train_if_needed()
    render_overview(messages, persona, selected_speaker)

    tabs = st.tabs(
        [
            "Persona JSON",
            "Dataset",
            "Persona Timeline",
            "Intent Classifier",
            "Memory Resolver",
            "System Design",
        ]
    )

    with tabs[0]:
        selected_speaker = render_persona_json_tab(persona)
    with tabs[1]:
        render_dataset_tab(messages)
    with tabs[2]:
        render_persona_tab(messages, persona, selected_speaker)
    with tabs[3]:
        render_intent_tab()
    with tabs[4]:
        render_resolver_tab(messages)
    with tabs[5]:
        render_design_tab()


if __name__ == "__main__":
    main()
