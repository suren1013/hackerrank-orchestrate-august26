"""Entry point for the Message Notification Router data pipeline.

Phase 1: Loads the datasets, prints a summary, and inspects one fully
         assembled message context.
Phase 2: Demonstrates the RetrievalEngine by printing a RetrievalResult
         for one sample message.
Phase 3: Demonstrates the MediaProcessor by printing MediaResult for one
         image message, one voice message, and one plain text message.
Phase 4: Demonstrates the AI Router by printing a complete routing
         decision (prompt summary, LLM output, validated output, final
         decision, evidence IDs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the code/ directory is on sys.path so absolute imports work
# regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR  # noqa: E402
from data.loader import DataLoader  # noqa: E402
from llm.router import Router  # noqa: E402
from media.processor import MediaProcessor  # noqa: E402
from retrieval.retriever import RetrievalEngine  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def print_dataset_summary(loader: DataLoader) -> None:
    """Print a clean summary of all loaded datasets."""
    summary = loader.summary()
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    if summary.empty:
        print("No datasets loaded.")
        return

    for _, row in summary.iterrows():
        print(f"\n[{row['dataset']}]")
        print(f"  Rows   : {row['rows']}")
        print(f"  Columns: {row['columns']}")

    if loader.missing_files:
        print("\n" + "-" * 80)
        print(f"WARNING: {len(loader.missing_files)} file(s) missing:")
        for f in loader.missing_files:
            print(f"  - {f}")
    print("=" * 80 + "\n")


def print_sample_context(contexts: list) -> None:
    """Print one fully assembled message context for inspection."""
    if not contexts:
        print("No contexts to inspect.")
        return

    ctx = contexts[0]
    print("=" * 80)
    print("SAMPLE MESSAGE CONTEXT (first message)")
    print("=" * 80)
    print(json.dumps(ctx.to_dict(), indent=2, default=str))
    print("=" * 80 + "\n")


def print_retrieval_result(result) -> None:
    """Print a RetrievalResult for inspection."""
    print("=" * 80)
    print("RETRIEVAL RESULT (first message)")
    print("=" * 80)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    print("=" * 80 + "\n")


def find_context_by_media_type(
    contexts: list, media_type: str
):
    """Return the first context with the given media_type."""
    for ctx in contexts:
        if ctx.media_type == media_type:
            return ctx
    return None


def print_media_results(contexts: list) -> None:
    """Demonstrate MediaProcessor on image, voice, and text messages."""
    media_processor = MediaProcessor()

    # 1. Image message.
    image_ctx = find_context_by_media_type(contexts, "image")
    if image_ctx is not None:
        result = media_processor.process(image_ctx)
        print("=" * 80)
        print(f"MEDIA RESULT (image) — {image_ctx.message_id}")
        print("=" * 80)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        print("=" * 80 + "\n")
    else:
        print("No image message found to demonstrate.\n")

    # 2. Voice message.
    voice_ctx = find_context_by_media_type(contexts, "voice")
    if voice_ctx is not None:
        result = media_processor.process(voice_ctx)
        print("=" * 80)
        print(f"MEDIA RESULT (voice) — {voice_ctx.message_id}")
        print("=" * 80)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        print("=" * 80 + "\n")
    else:
        print("No voice message found to demonstrate.\n")

    # 3. Plain text message (no media).
    text_ctx = None
    for ctx in contexts:
        if not ctx.media_type:
            text_ctx = ctx
            break
    if text_ctx is not None:
        result = media_processor.process(text_ctx)
        print("=" * 80)
        print(f"MEDIA RESULT (text, no media) — {text_ctx.message_id}")
        print("=" * 80)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        print("=" * 80 + "\n")
    else:
        print("No text message found to demonstrate.\n")


def print_routing_decision(trace) -> None:
    """Print a complete routing decision trace for inspection."""
    print("=" * 80)
    print(f"ROUTING DECISION — {trace.message_id}")
    print("=" * 80)

    print("\n--- PROMPT SUMMARY ---")
    print(trace.prompt_summary)

    print("\n--- LLM OUTPUT ---")
    print(json.dumps(trace.llm_response.to_dict(), indent=2, default=str))

    print("\n--- VALIDATED OUTPUT ---")
    print(json.dumps(trace.validated.to_dict(), indent=2, default=str))

    print("\n--- CALIBRATION NOTES ---")
    for note in trace.calibration_notes:
        print(f"  - {note}")

    print("\n--- FINAL DECISION ---")
    print(json.dumps(trace.final_decision.to_dict(), indent=2, default=str))

    print("\n--- OUTPUT ROW (output.csv contract) ---")
    print(json.dumps(trace.final_decision.to_output_row(), indent=2, default=str))
    print("=" * 80 + "\n")


def main() -> None:
    """Run the full pipeline: load, retrieve, process media, and route."""
    logger.info("Starting Message Notification Router data pipeline.")

    # 1. Load all datasets.
    loader = DataLoader()

    # 2. Print dataset summary.
    print_dataset_summary(loader)

    # 3. Assemble unified message contexts.
    contexts = loader.build_contexts()

    # 4. Print one fully assembled context for inspection.
    print_sample_context(contexts)

    # 5. Demonstrate the RetrievalEngine on the first message.
    if contexts:
        engine = RetrievalEngine()
        result = engine.retrieve(contexts[0])
        print_retrieval_result(result)

    # 6. Demonstrate the MediaProcessor (image, voice, text).
    print_media_results(contexts)

    # 7. Demonstrate the AI Router on the first message.
    if contexts:
        router = Router()
        trace = router.route(contexts[0])
        print_routing_decision(trace)

    # 8. Ensure the output directory exists for later phases.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", OUTPUT_DIR)

    logger.info("Data pipeline completed successfully.")


if __name__ == "__main__":
    main()