# Self-Evaluation

| Requirement | Status | Notes |
| --- | --- | --- |
| Round 1 persona JSON | Complete | Uses the provided persona extraction JSON at data/round1_persona.json |
| Round 1 baseline extraction | Complete | Extracts speaker traits and communication style from the persona JSON |
| Topic splitting | Complete | Detects topic checkpoints for user messages and displays them in the Dataset tab |
| Persona drift detector | Complete | Groups user messages by day and detects day-to-day tone changes |
| Day-wise timeline | Complete | Shows date, day label, persona state, drift flag, and trigger |
| Trigger detection | Complete | Detects topics, events, and people from message text |
| Offline intent classifier | Complete | Uses local scikit-learn model with no API calls |
| Model under 50MB | Complete | Current model is about 0.043 MB |
| CPU inference under 200ms | Complete | Local benchmark is about 1 ms per message |
| Conflict-aware retrieval | Complete | Ranks chunks by recency, emotional weight, and keyword match |
| Contradiction flagging | Complete | Flags positive/negative context changes about the same topic |
| System design doc | Complete | Covers local storage, sync boundaries, and conflict strategy |
| Hosted demo | Pending | To be deployed after local testing |
| Loom walkthrough | Pending | To be recorded after final demo is ready |

## Test Evidence

```text
Model size: 0.043 MB
Average benchmark latency: about 1 ms
Sister query retrieved chunks: 3
Contradictions flagged: yes
```

## Known Limitations

- The conversation dataset is synthetic because no official dataset was provided.
- Persona detection uses explainable keyword scoring rather than a large language model.
- The resolver handles the required contradiction pattern but is not a full natural language inference system.
