# Submission Checklist

## Pre-Submission Verification

- [x] **output.csv generated** — `output/output.csv` with 110 rows + header
- [x] **traces generated** — `output/traces/` with per-message JSON traces
- [x] **evaluation completed** — `output/evaluation/` with metrics, plots, reports
- [x] **README updated** — `code/README.md` reflects current implementation
- [x] **requirements verified** — `code/requirements.txt` matches imported packages
- [x] **.env.example verified** — All `os.getenv()` variables documented
- [x] **API keys excluded** — `.env` in `.gitignore`, `.env.example` has placeholder values
- [x] **no temporary files committed** — Test scripts removed, `__pycache__` gitignored
- [x] **.gitignore verified** — Ignores `.env`, `code/output/`, `__pycache__/`, `*.log`
- [x] **pipeline reproducible** — `python main.py` processes all 110 messages deterministically
- [x] **CLI verified** — `--single`, `--resume`, `--trace`, `--evaluate` all work

## Output Contract Verification

- [x] Columns: `message_id,action,message_type,reason,confidence,evidence_message_ids`
- [x] One row per message_id in `dataset/messages.csv`
- [x] Confidence rounded to 4 decimals
- [x] Evidence IDs semicolon-separated
- [x] `none` when no evidence exists
- [x] action ∈ {notify, digest, mute}
- [x] message_type ∈ allowed values

## Documentation Verification

- [x] `README.md` — Project overview, installation, CLI, output format
- [x] `ARCHITECTURE.md` — Module details with Mermaid diagram
- [x] `SYSTEM_DESIGN.md` — Design decisions, scalability, failure handling
- [x] `PROJECT_STRUCTURE.md` — Auto-generated directory tree
- [x] `docs/overview.md` — Quick summary
- [x] `docs/architecture.md` — Architecture cross-reference
- [x] `docs/system_design.md` — Design cross-reference
- [x] `docs/prompt_strategy.md` — Prompt evolution and rationale
- [x] `docs/evaluation.md` — Evaluation framework description
- [x] `docs/submission_checklist.md` — This file
- [x] `docs/ai_judge_questions.md` — Anticipated judge questions
- [x] `docs/presentation_outline.md` — Presentation structure

## Files to Submit

1. **code.zip** — Full runnable solution (code/ directory)
2. **output.csv** — Predictions for all 110 messages
3. **chat_transcript** — log.txt from `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt`

## Final Commands

```bash
# Verify output.csv
python main.py --single msg_023

# Verify trace inspection
python main.py --trace msg_023

# Run evaluation
python main.py --evaluate

# Full pipeline (if needed)
python main.py