# System Design

See [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) for the full system design document covering design goals, decisions, scalability, failure handling, retry strategy, and resume strategy.

## Key Design Principles

1. **Retrieval-first**: Personalized evidence before any LLM call
2. **Policy before LLM**: Deterministic rules for clear-cut cases; LLM only for ambiguous messages
3. **Provider abstraction**: Switch models via environment variables
4. **Confidence calibration**: Combine LLM + deterministic signals
5. **Production-ready**: Checkpointing, resume, traces, statistics

Cross-references: [Architecture](architecture.md) | [Prompt Strategy](prompt_strategy.md) | [Overview](overview.md)