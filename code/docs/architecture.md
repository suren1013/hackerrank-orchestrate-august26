# Architecture

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full architecture document with Mermaid diagram and per-module details.

## Quick Reference

| Module | Location | Purpose |
|---|---|---|
| DataLoader | `data/loader.py` | Load CSVs, build MessageContext |
| RetrievalEngine | `retrieval/retriever.py` | Personalized evidence retrieval |
| MediaProcessor | `media/processor.py` | OCR + Whisper + text analysis |
| PolicyEngine | `policy/engine.py` | Deterministic rules before LLM |
| Router | `llm/router.py` | LLM routing for ambiguous messages |
| ConfidenceCalibration | `llm/confidence.py` | Combine LLM + deterministic signals |
| InferencePipeline | `inference/pipeline.py` | Production orchestration |
| Evaluator | `evaluation/evaluator.py` | Read-only evaluation suite |

Cross-references: [Overview](overview.md) | [System Design](system_design.md) | [Prompt Strategy](prompt_strategy.md)