# System Design

## Goal

The system is local-first. Raw conversations stay on the user's device, while compact memory checkpoints can sync across devices.

## Components

- Local app: loads the Round 1 persona JSON and runs persona drift detection, offline intent classification, and memory retrieval.
- Local storage: keeps raw conversations, model files, and checkpoint metadata.
- Sync service: receives summaries and metadata, not complete private logs.
- Resolver: merges local and synced checkpoints while preserving conflicts.

## On-Device Storage

The device stores raw messages, local intent predictions, persona states, topic checkpoints, and retrieval metadata. The offline intent model also runs on the device, so classification does not require an external API.

## What Syncs

Only compact memory data syncs:

- topic checkpoint id
- short summary
- timestamp
- mood and tone labels
- emotional weight
- conflict flags

## What Stays Local

Raw conversations, sensitive entities, and full emotional logs stay local unless the user chooses to export or sync them.

## Conflict Resolution

Simple metadata can use last-write-wins. Memory content should not be overwritten silently. When two checkpoints disagree, the system preserves both, ranks them by recency and emotional weight, and marks the answer as having conflicting context. This is safer for personal memory because an older emotional event may still matter even if a newer message changes the current state.

## Diagram

```text
+-----------------------+
| User Device           |
| raw messages local    |
| offline classifier    |
+-----------+-----------+
            |
            v
+-----------------------+
| Local Memory Engine   |
| drift + RAG resolver  |
+-----------+-----------+
            |
            | summaries + metadata
            v
+-----------------------+
| Sync Server           |
| checkpoints only      |
+-----------+-----------+
            |
            v
+-----------------------+
| Second Device         |
| rebuilds local view   |
+-----------------------+
```

## Tradeoffs

Keeping raw data local improves privacy but makes cross-device search harder. Syncing summaries is lighter and safer, but summaries may lose detail. Preserving conflicting memories is more honest than overwriting them, but the UI must explain uncertainty clearly.
