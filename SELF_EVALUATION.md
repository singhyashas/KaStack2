# Self-Evaluation

| Requirement | Status | Notes |
| --- | --- | --- |
| Round 1 persona JSON | Complete | Uses the provided persona extraction JSON at data/round1_persona.json |
| Round 1 baseline extraction | Complete | Extracts speaker traits and communication style from the persona JSON |
| Topic splitting | Complete | Detects topic checkpoints for user messages and displays them in the Dataset tab |
| Persona drift detector | Complete | Groups user messages by day and detects day-to-day tone changes |
| Day-wise timeline | Complete | Shows date, day label, persona state, drift flag, and trigger |
| Trigger detection | Complete | Detects topics, events, and people from message text |
| Offline intent classifier | Complete | Uses a local pure-Python model with no API calls |
| Model under 50MB | Complete | Current model is about 0.005 MB |
| CPU inference under 200ms | Complete | Local benchmark is well under 200 ms per message |
| Conflict-aware retrieval | Complete | Ranks chunks by recency, emotional weight, and keyword match |
| Contradiction flagging | Complete | Flags positive/negative context changes about the same topic |
| System design doc | Complete | Covers local storage, sync boundaries, and conflict strategy |
| GitHub repo | Complete | https://github.com/singhyashas/KaStack2 |
| Hosted demo | Complete | https://kastack2-uswnirmxb9evozzssbbucc.streamlit.app/ |
| Loom walkthrough | Complete | https://www.loom.com/share/67470d2d4b9a48979e115b5ba90bc652 |

## Test Evidence

```text
Model size: about 0.005 MB
Average benchmark latency: well under 200 ms
Sister query retrieved chunks: 3
Contradictions flagged: yes
```

## Known Limitations

- The conversation dataset is synthetic because no official dataset was provided.
- Persona detection uses explainable keyword scoring rather than a large language model.
- The resolver handles the required contradiction pattern but is not a full natural language inference system.
