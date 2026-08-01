# System Design

## Design Goals

1. **Minimize unnecessary interruptions** while never missing genuinely important messages
2. **Personalized decisions** using the full context (user, sender, business, group, history, media)
3. **Deterministic where possible** — scams and trusted urgent messages don't need LLM reasoning
4. **Provider-independent** — switch LLM models via environment variables, not code changes
5. **Production-ready** — checkpointing, resume, error handling, traces, statistics
6. **Evaluable** — systematic metrics, confusion matrices, error analysis, threshold recommendations

## Design Decisions

### Retrieval-First Reasoning

The system retrieves personalized evidence *before* any LLM call. This means:
- The LLM receives a compact, relevant context instead of raw data dumps
- Evidence IDs are deterministic and traceable (no hallucinated evidence)
- Retrieval quality feeds into confidence calibration

**Why**: LLMs reason better over curated context than over raw data. Retrieval also enables deterministic policy rules to fire before the LLM, saving API calls and improving reliability.

### Policy Executes Before the LLM

The PolicyEngine runs after retrieval and media processing but before the LLM:
- **Scam detection**: OTP/PIN/phishing + low trust → mute (confidence 0.92)
- **Trusted urgent**: Family/admin/verified bank + urgency keywords → notify (confidence 0.88)
- **Media failure**: Failed OCR/ASR → digest (confidence 0.55)
- **Repetitive spam**: High forward count + high ignore rate → mute (confidence 0.82)

**Why**: These patterns are deterministic and don't benefit from LLM reasoning. Running them first saves API costs, reduces latency, and improves reliability for clear-cut cases. The LLM only processes ambiguous messages.

### Confidence Calibration

The LLM's raw confidence is combined with:
- Retrieval quality (20% blend of average evidence similarity)
- Safety signal strength (boosts mute confidence when trust is low)
- Media extraction confidence (10% blend)
- Trust/interest score anchors

**Why**: LLM confidence alone is poorly calibrated. Combining it with deterministic signals produces confidence that better correlates with correctness. The calibration gap (avg confidence of correct vs incorrect predictions) is tracked in evaluation.

### Provider Abstraction

```
LLMProvider (ABC)
├── OpenAIProvider   — OpenAI-compatible API
├── GeminiProvider   — Google Gemini with retry + validation
├── OllamaProvider   — Local Ollama
└── MockProvider     — Deterministic fallback (no API key)
```

**Why**: The AI Judge values clean architecture. Provider abstraction makes the design easy to explain and lets us switch models by changing `LLM_PROVIDER` env var. Each provider has retry logic with exponential backoff (1s → 2s → 4s, max 3 retries) for transient failures (429, 500, 502, 503, 504).

### Prompt Engineering

The system prompt evolved through multiple phases:
- **Phase 4**: Basic JSON output instructions
- **Phase 6**: Strict JSON-only output, no markdown, no code fences
- **Phase 6.5**: Renamed ambiguous fields (Trust Score → Overall Notification Priority Score, Business Trust → Business Verification Score), natural-language retrieval summaries, structured prompt sections
- **Phase 6.6**: Decision Guide, Example Decision Matrix, Score Explanations, DO NOT rules, Confidence Guide, 6 few-shot examples

**Why**: The prompt directly addresses observed confusion between sender authenticity and notification priority. Structured sections (USER PROFILE, BUSINESS CONTEXT, PRIORITY SIGNALS, CURRENT MESSAGE, SIMILAR HISTORY) make the context easy for the LLM to follow.

### Trace Logging

Every message gets a JSON trace in `output/traces/msg_xxx.json` containing:
- retrieval_summary, media_summary, policy_used, policy_rule
- prompt_summary, llm_response, confidence_before/after
- final_decision, output_row, latencies

**Why**: Traces enable debugging, evaluation enrichment, and AI Judge inspection. No API keys or secrets are stored in traces.

## Sequence Diagram

```
Message → DataLoader.build_contexts()
         → RetrievalEngine.retrieve(context)
         → MediaProcessor.process(context)
         → PolicyEngine.evaluate(context, retrieval, media)
            ├─ PolicyDecision → RouterTrace (LLM skipped)
            └─ None → Router.route(context, retrieval, media)
                       → PromptBuilder.build_user_prompt()
                       → LLMProvider.complete(system, user)
                       → Validator.validate_response()
                       → ConfidenceCalibration.calibrate()
                       → Postprocessor.build_final_decision()
                       → RouterTrace
         → OutputWriter.write(rows)
         → TraceLogger.save_trace(msg_id, trace)
```

## Scalability

- **Retrieval**: O(n) per message over historical messages — efficient for hundreds
- **Media**: Disk caching (keyed by media_id + file mtime) + LRU in-memory cache
- **LLM**: Only called for ambiguous messages (~55% of messages use policy instead)
- **Checkpointing**: Every 10 messages, enabling resume after interruptions
- **Batch processing**: tqdm progress bar, per-message error handling (failures don't stop the pipeline)

## Failure Handling

- **Per-message failures**: Logged, recorded in statistics, processing continues
- **LLM failures**: Retry with exponential backoff, then fallback to digest/unknown
- **Media failures**: Policy rule digests the message; OCR/ASR errors are cached as non-errors
- **Validation failures**: Safe fallback (digest/unknown/0.5)
- **Checkpoint**: partial_output.csv enables resume after crash

## Retry Strategy

- HTTP 429/500/502/503/504: Retry up to 3 times with exponential backoff (1s, 2s, 4s)
- Connection errors and read timeouts: Same retry strategy
- After retries exhausted: Exception raised, router falls back to safe default

## Resume Strategy

1. `python main.py --resume` loads processed message IDs from `output/partial_output.csv`
2. Already-processed messages are skipped
3. New messages are processed and appended
4. Final output.csv is written with all rows (existing + new)
5. Checkpoint cleared on successful completion

## Future Improvements

- Vector-based retrieval (embeddings) for larger datasets
- Batch LLM calls for throughput optimization
- Additional policy rules (quiet hours, opt-out enforcement, sender reputation)
- A/B testing framework for prompt variations
- Fine-tuning thresholds based on evaluation results
- Additional LLM providers (Anthropic Claude, Mistral)