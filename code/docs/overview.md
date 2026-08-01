# Overview

See [README.md](../README.md) for the full project overview. This file provides a quick summary.

## What This System Does

The Message Notification Router classifies every incoming WhatsApp message as:
- **notify**: interrupt the user now
- **digest**: show later
- **mute**: suppress as low-value, repetitive, or unsafe

## How It Works

1. **DataLoader** loads 11 CSVs and builds a unified `MessageContext` per message
2. **RetrievalEngine** finds personalized evidence from historical messages
3. **MediaProcessor** runs OCR (images) and Whisper (voice notes)
4. **PolicyEngine** applies deterministic rules (scam detection, trusted urgent, etc.)
5. **Router** calls the LLM only for ambiguous messages
6. **ConfidenceCalibration** combines LLM + deterministic signals
7. **OutputWriter** writes `output.csv` with the exact HackerRank contract

## Key Numbers

- 110 messages processed
- 49 policy decisions (LLM skipped)
- 61 LLM decisions
- 100% success rate
- ~1.15s average per message

Cross-references: [Architecture](../ARCHITECTURE.md) | [System Design](../SYSTEM_DESIGN.md) | [Prompt Strategy](prompt_strategy.md) | [Evaluation](evaluation.md)