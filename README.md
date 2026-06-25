# Adaptive Persona Memory Engine

A compact Streamlit project for persona drift detection, offline intent classification, and conflict-aware memory retrieval.

## Current Results

- Persona timeline includes Day 1 as curious & formal, Day 4 as casual & frustrated, and Day 7 as casual & playful.
- Round 1 persona input is included at `data/round1_persona.json` using the provided extraction schema.
- The app summarizes Round 1 speaker traits, facts, preferences, and communication style before showing drift.
- Detected topic checkpoints are shown separately from the stored sample topics.
- Intent model size after training: about 0.043 MB.
- Benchmark inference latency: about 1 ms per message on local CPU.
- The sister memory query retrieves three relevant chunks, ranks the latest memory highest, and flags the contradiction.

## Features

- Day-wise persona timeline from multi-day conversation logs
- Round 1 persona JSON viewer
- Round 1 baseline extraction from speaker traits and communication style
- Topic checkpoint splitting for user messages
- Mood and tone drift detector with likely trigger extraction
- Offline intent classifier for reminder, emotional-support, action-item, small-talk, and unknown
- CPU-friendly scikit-learn model with a small disk footprint
- Conflict-aware memory resolver for contradictory memories
- One-page system design and self-evaluation files

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- joblib

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Main Demo Cases

Use these in the walkthrough:

- Persona Timeline tab: show the day-wise drift table and trigger column.
- Dataset tab: show detected topic checkpoints.
- Intent Classifier tab: try "Remind me to call my sister tomorrow evening".
- Intent Classifier tab: try "I feel overwhelmed and need support".
- Memory Resolver tab: ask "Did I mention anything about my sister?"

## Dataset

The Round 1 persona JSON is loaded from `data/round1_persona.json`. No official multi-day drift or intent dataset was provided with the task, so this repo includes a small synthetic conversation dataset designed around the required cases: multi-day tone shifts, intent examples, and contradictory sister-related memories.

## Project Flow

```text
Round 1 Persona JSON + Conversation JSON
      |
      v
Persona Engine
  baseline extraction, day grouping, tone scoring, trigger detection
      |
      v
Offline Intent Classifier
  TF-IDF + Logistic Regression
      |
      v
Memory Resolver
  retrieval, recency scoring, emotional weight, contradiction flagging
```

## Model Choice

The intent classifier uses TF-IDF features and Logistic Regression. This keeps inference local, fast, and small enough for the placement requirement of a lightweight offline model.

## Conflict Resolver

The resolver retrieves chunks that match the query subject, then ranks them with:

```text
final_score = 0.55 * recency + 0.35 * emotional_weight + 0.10 * keyword_match
```

For the sister query, the resolver keeps all three memories instead of dropping older context. It marks the older tension and newer improvement as a contradiction and returns a merged answer.

## Loom Walkthrough Outline

1. Open the app and show the four requirement metrics.
2. Open the Persona JSON tab and show the Round 1 speaker summary.
3. Explain the sample dataset and topic split.
4. Show the persona timeline and drift triggers.
5. Run the offline intent classifier and point out model size and latency.
6. Ask the sister question in the memory resolver.
7. Open the system design tab and explain what syncs versus what stays local.

## Limitations

- The multi-day conversation and intent training datasets are synthetic because no official dataset was supplied for those modules.
- Persona detection is explainable and rule-based, so it favors clarity over deep language nuance.
- Contradiction detection handles common status and sentiment conflicts, not every possible linguistic contradiction.
