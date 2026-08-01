# Project Structure

Auto-generated from the actual repository contents.

```
code/
├── .env.example               # Environment variable template
├── .env                       # Local environment (gitignored)
├── ARCHITECTURE.md            # Detailed architecture document
├── README.md                  # Project overview and instructions
├── SYSTEM_DESIGN.md           # System design rationale
├── PROJECT_STRUCTURE.md       # This file
├── config.py                  # Central configuration with env var overrides
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── data/
│   ├── __init__.py
│   └── loader.py              # DataLoader: loads CSVs, builds MessageContext
├── evaluation/
│   ├── __init__.py
│   ├── analyzer.py            # Distribution, error, confidence, latency analysis
│   ├── confusion.py           # Confusion matrix generation
│   ├── evaluator.py           # Evaluator orchestrator (read-only)
│   ├── main.py                # (legacy placeholder)
│   ├── metrics.py             # Accuracy, precision, recall, F1
│   ├── plots.py               # matplotlib chart generation
│   ├── report.py              # Markdown + HTML report generation
│   └── thresholds.py          # Threshold recommendations
├── inference/
│   ├── __init__.py
│   ├── checkpoint.py          # CheckpointManager (partial_output.csv, resume)
│   ├── pipeline.py            # InferencePipeline orchestrator
│   ├── statistics.py          # PipelineStatistics (latency, success/failure)
│   ├── trace_logger.py        # TraceLogger (per-message JSON traces)
│   └── writer.py              # OutputWriter (output.csv)
├── llm/
│   ├── __init__.py
│   ├── confidence.py          # Confidence calibration
│   ├── postprocessor.py       # Evidence ID attachment (from retrieval only)
│   ├── prompt_builder.py      # Structured prompt builder
│   ├── prompts.py             # Legacy prompt templates
│   ├── providers.py           # LLMProvider ABC + OpenAI/Gemini/Ollama/Mock
│   ├── router.py              # Router orchestrator (policy + LLM)
│   ├── schemas.py             # LLMResponse, RoutingDecision, RouterTrace
│   └── validator.py           # Output validation
├── media/
│   ├── __init__.py
│   ├── audio_processor.py     # faster-whisper voice transcription
│   ├── image_processor.py     # Tesseract OCR
│   ├── processor.py           # MediaProcessor orchestrator + disk cache
│   ├── text_analysis.py       # Entity/safety/urgency detection
│   └── types.py               # MediaResult dataclass
├── models/
│   ├── __init__.py
│   └── schemas.py             # MessageContext dataclass
├── policy/
│   ├── __init__.py
│   ├── engine.py              # PolicyEngine orchestrator
│   ├── rules.py               # Deterministic policy rules
│   └── types.py               # PolicyDecision dataclass
├── retrieval/
│   ├── __init__.py
│   ├── analyzers.py           # Sender/business/group/engagement analysis
│   ├── evidence.py            # Evidence ranking
│   ├── retriever.py           # RetrievalEngine orchestrator
│   ├── similarity.py          # RapidFuzz similar message retrieval
│   ├── summarizer.py          # Natural-language retrieval summary
│   └── types.py               # RetrievalResult and related dataclasses
├── utils/
│   ├── __init__.py
│   ├── helpers.py             # is_empty, safe_str, normalize_text, etc.
│   └── logger.py              # Centralized logging setup
└── output/                    # Generated artifacts (gitignored)
    ├── output.csv             # Final predictions
    ├── partial_output.csv     # Checkpoint (transient)
    ├── traces/                # Per-message JSON traces
    └── evaluation/            # Evaluation reports, plots, metrics