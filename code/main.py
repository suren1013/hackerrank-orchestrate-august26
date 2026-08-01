"""Entry point for the Message Notification Router data pipeline.

This phase only loads the datasets, prints a summary, and inspects one fully
assembled message context. No routing, OCR, or ASR happens yet.
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


def main() -> None:
    """Run the data pipeline: load, summarize, and inspect one context."""
    logger.info("Starting Message Notification Router data pipeline.")

    # 1. Load all datasets.
    loader = DataLoader()

    # 2. Print dataset summary.
    print_dataset_summary(loader)

    # 3. Assemble unified message contexts.
    contexts = loader.build_contexts()

    # 4. Print one fully assembled context for inspection.
    print_sample_context(contexts)

    # 5. Ensure the output directory exists for later phases.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", OUTPUT_DIR)

    logger.info("Data pipeline completed successfully.")


if __name__ == "__main__":
    main()