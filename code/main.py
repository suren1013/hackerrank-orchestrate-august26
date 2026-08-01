"""Entry point for the Message Notification Router.

CLI modes:
    python main.py                  Process the entire dataset -> output/output.csv
    python main.py --single msg_023 Process one message only
    python main.py --resume         Resume from partial_output.csv checkpoint
    python main.py --trace msg_023  Pretty-print a saved trace for a message

The InferencePipeline orchestrates all existing modules (DataLoader,
RetrievalEngine, MediaProcessor, PolicyEngine, Router) without duplicating
any of their logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the code/ directory is on sys.path so absolute imports work
# regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR  # noqa: E402
from data.loader import DataLoader  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from inference.pipeline import InferencePipeline  # noqa: E402
from inference.trace_logger import TraceLogger  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Run the production inference pipeline with CLI options."""
    parser = argparse.ArgumentParser(
        description="Message Notification Router — Production Inference Pipeline"
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Process only this message_id.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial_output.csv checkpoint.",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Pretty-print the saved trace for this message_id and exit.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the evaluation suite against ground truth and exit.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth CSV (default: dataset/sample_messages.csv).",
    )
    args = parser.parse_args()

    # ---- Trace inspection mode --------------------------------------
    if args.trace:
        trace_logger = TraceLogger()
        trace_logger.pretty_print_trace(args.trace)
        return

    # ---- Evaluation mode --------------------------------------------
    if args.evaluate:
        evaluator = Evaluator()
        gt_path = args.ground_truth if args.ground_truth else None
        evaluator.evaluate(ground_truth_path=gt_path)
        return

    # ---- Inference mode ----------------------------------------------
    logger.info("Starting Message Notification Router inference pipeline.")
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Instantiate the pipeline (loads datasets automatically).
    pipeline = InferencePipeline(loader=DataLoader())

    # Run the pipeline with the requested options.
    rows = pipeline.run(resume=args.resume, single=args.single)

    if args.single:
        if rows:
            logger.info("Printed decision for %s.", args.single)
        else:
            logger.warning("No output produced for %s.", args.single)
    else:
        logger.info(
            "Inference complete: %d rows written to %s.",
            len(rows),
            pipeline.writer.output_path,
        )


if __name__ == "__main__":
    main()