# Message Notification Router

AI-powered WhatsApp message notification routing system for the HackerRank Orchestrate August 2026 hackathon.

## Project Overview

The Message Notification Router decides for every incoming WhatsApp message whether the user should be **notified** now, whether the message can be **digested** for later, or whether it should be **muted** as low-value, repetitive, unwanted, suspicious, or unsafe.

The system reasons over multimodal messages (text, image posters/screenshots, voice notes) using personalized context from user profiles, group metadata, business relationships, historical engagement, and media analysis.

## Problem Statement

WhatsApp is noisy. A user receives family chats, society notices, school updates, co-worker messages, business promotions, image posters, voice notes, and scams in the same stream. Treating every message the same creates two bad outcomes: important messages get missed, and unwanted messages interrupt the user.

The router must produce `output.csv` with one prediction per message in `dataset/messages.csv`:

```
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `action`: `notify` | `digest` | `mute`
- `message_type`: `personal` | `urgent` | `event` | `payment` | `business_update` | `promotion` | `greeting` | `forward` | `spam` | `scam` | `unknown`
- `confidence`: 0.0 to 1.0
- `evidence_message_ids`: semicolon-separated historical message IDs, or `none`

## Key Features

- **Retrieval-first reasoning**: Personalized evidence from historical messages, sender/business/group relationships, and engagement patterns
- **Multimodal intelligence**: Tesseract OCR for images, faster-whisper for voice transcription, deterministic entity/safety/urgency detection
- **Deterministic policy engine**: Scam detection, trusted-urgent detection, media failure handling, and repetitive spam filtering — all before the LLM
- **Provider abstraction**: Switch between Gemini, OpenAI, Ollama, or Mock provider via environment variables
- **Confidence calibration**: Combines LLM confidence with retrieval quality, safety signals, media confidence, and trust/interest scores
- **Production inference pipeline**: Checkpointing, resume, progress bar, per-message traces, statistics report
- **Evaluation suite**: Metrics, confusion matrices, distribution analysis, error analysis, confidence calibration, threshold recommendations, plots

## High-Level Architecture

```
MessageContext → RetrievalEngine → MediaProcessor → PolicyEngine → Router (LLM) → ConfidenceCalibration → OutputWriter
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details and [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for design rationale.

## Installation

```bash
# Clone the repository
git clone https://github.com/suren1013/hackerrank-orchestrate-august26.git
cd hackerrank-orchestrate-august26/code

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR (Windows)
winget install --id UB-Mannheim.TesseractOCR

# Copy environment template
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Requirements

- Python 3.10+
- Tesseract OCR binary (for image OCR)
- See `requirements.txt` for Python packages

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | Provider: `gemini`, `openai`, `ollama`, `mock` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OPENAI_BASE_URL` | — | Custom OpenAI-compatible base URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `LOG_LEVEL` | `INFO` | Logging level |
| `WHISPER_DEVICE` | `cpu` | Whisper device: `cpu` or `cuda` |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model size |
| `TESSERACT_CMD` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Tesseract binary path |

## Running the Pipeline

### Process the entire dataset

```bash
python main.py
```

Generates `output/output.csv` with one prediction per message.

### Process a single message

```bash
python main.py --single msg_023
```

### Resume from checkpoint

```bash
python main.py --resume
```

Skips already-processed message IDs from `output/partial_output.csv`.

### Inspect a trace

```bash
python main.py --trace msg_023
```

Pretty-prints the saved JSON trace for a message.

### Run evaluation

```bash
python main.py --evaluate
python main.py --evaluate --ground-truth path/to/ground_truth.csv
```

Generates metrics, confusion matrices, plots, and reports in `output/evaluation/`.

## Output Format

`output/output.csv` with exact columns:

```csv
message_id,action,message_type,reason,confidence,evidence_message_ids
msg_023,digest,payment,Routine banking update from a known business that can wait.,0.8627,message_0243;message_0102;message_0101;message_0011;message_0242
```

- Confidence rounded to 4 decimals
- Evidence IDs semicolon-separated; `none` when empty

## Project Structure

```
code/
├── main.py                    # CLI entry point
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── data/                      # Data loading & context assembly
│   └── loader.py
├── retrieval/                 # Personalized retrieval engine
│   ├── retriever.py
│   ├── analyzers.py
│   ├── similarity.py
│   ├── evidence.py
│   ├── summarizer.py
│   └── types.py
├── media/                     # Multimodal intelligence
│   ├── processor.py
│   ├── image_processor.py
│   ├── audio_processor.py
│   ├── text_analysis.py
│   └── types.py
├── policy/                    # Deterministic policy engine
│   ├── engine.py
│   ├── rules.py
│   └── types.py
├── llm/                       # LLM routing layer
│   ├── router.py
│   ├── providers.py
│   ├── prompt_builder.py
│   ├── validator.py
│   ├── confidence.py
│   ├── postprocessor.py
│   └── schemas.py
├── inference/                 # Production inference pipeline
│   ├── pipeline.py
│   ├── writer.py
│   ├── checkpoint.py
│   ├── statistics.py
│   └── trace_logger.py
├── evaluation/                # Evaluation & model tuning
│   ├── evaluator.py
│   ├── metrics.py
│   ├── confusion.py
│   ├── analyzer.py
│   ├── thresholds.py
│   ├── report.py
│   └── plots.py
├── models/                    # Data schemas
│   └── schemas.py
├── utils/                     # Shared utilities
│   ├── logger.py
│   └── helpers.py
└── output/                    # Generated artifacts
    ├── output.csv
    ├── partial_output.csv     # Checkpoint
    ├── traces/                # Per-message JSON traces
    └── evaluation/            # Evaluation reports & plots
```

## Future Improvements

- Fine-tune policy thresholds based on evaluation results
- Add more policy rules (e.g., opt-out detection, quiet hours enforcement)
- Implement batch LLM calls for throughput
- Add vector-based retrieval (embeddings) for larger datasets
- Support additional LLM providers (Anthropic Claude, Mistral)
- Add A/B testing framework for prompt variations