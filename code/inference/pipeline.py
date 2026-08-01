"""InferencePipeline: orchestrates the full production inference flow.

Message Context
    ↓
Retrieval Engine
    ↓
Media Processor
    ↓
Policy Engine
    ↓
LLM Router (only if policy does not override)
    ↓
Confidence Calibration
    ↓
Output Writer

Reuses all existing modules — no duplicated logic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data.loader import DataLoader
from inference.checkpoint import CheckpointManager
from inference.statistics import PipelineStatistics
from inference.trace_logger import TraceLogger
from inference.writer import OutputWriter
from llm.router import Router
from media.processor import MediaProcessor
from models.schemas import MessageContext
from policy.engine import PolicyEngine
from retrieval.retriever import RetrievalEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """Production inference pipeline for the Message Notification Router."""

    def __init__(
        self,
        loader: DataLoader | None = None,
        retrieval_engine: RetrievalEngine | None = None,
        media_processor: MediaProcessor | None = None,
        policy_engine: PolicyEngine | None = None,
        router: Router | None = None,
        writer: OutputWriter | None = None,
        checkpoint: CheckpointManager | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        self.loader = loader or DataLoader()
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.media_processor = media_processor or MediaProcessor()
        self.policy_engine = policy_engine or PolicyEngine()
        self.router = router or Router(
            retrieval_engine=self.retrieval_engine,
            media_processor=self.media_processor,
            policy_engine=self.policy_engine,
        )
        self.writer = writer or OutputWriter()
        self.checkpoint = checkpoint or CheckpointManager()
        self.trace_logger = trace_logger or TraceLogger()
        self.stats = PipelineStatistics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        resume: bool = False,
        single: str | None = None,
    ) -> list[dict]:
        """Run the full inference pipeline.

        Parameters
        ----------
        resume : bool
            If True, skip message IDs already in the checkpoint.
        single : str | None
            If set, process only this message ID.

        Returns
        -------
        list[dict]
            List of output rows.
        """
        logger.info("Pipeline started.")

        # Load contexts.
        contexts = self.loader.build_contexts()
        if single:
            contexts = [c for c in contexts if c.message_id == single]
            if not contexts:
                logger.warning("Message %s not found.", single)
                return []

        # Load checkpoint for resume.
        processed_ids: set[str] = set()
        if resume:
            processed_ids = self.checkpoint.load_processed_ids()
            logger.info("Resuming: %d messages already processed.", len(processed_ids))

        # Filter out already-processed messages.
        to_process = [c for c in contexts if c.message_id not in processed_ids]
        logger.info("Processing %d messages.", len(to_process))

        output_rows: list[dict] = []
        if resume:
            # Load existing rows from checkpoint.
            output_rows = self._load_checkpoint_rows()

        # Process with progress bar.
        for ctx in tqdm(to_process, desc="Processing Messages", unit="msg"):
            try:
                row, trace = self._process_one(ctx)
                output_rows.append(row)
                self.trace_logger.save_trace(ctx.message_id, trace)
            except Exception as exc:
                logger.error("Failed to process %s: %s", ctx.message_id, exc)
                self.stats.record_failure(ctx.message_id)

            # Checkpoint every N messages.
            if len(output_rows) % self.checkpoint.interval == 0:
                self.checkpoint.save_checkpoint(output_rows)

        # Write final output.
        self.writer.write(output_rows)

        # Clear checkpoint on success.
        if not self.stats.failed_messages:
            self.checkpoint.clear_checkpoint()

        # Finish statistics and print report.
        self.stats.finish()
        self.stats.print_report()

        logger.info("Pipeline finished. Output: %s", self.writer.output_path)
        return output_rows

    # ------------------------------------------------------------------
    # Single-message processing
    # ------------------------------------------------------------------

    def _process_one(
        self, ctx: MessageContext
    ) -> tuple[dict, dict[str, Any]]:
        """Process one message context and return (output_row, trace)."""
        start = time.time()

        # 1. Retrieval.
        t0 = time.time()
        retrieval = self.retrieval_engine.retrieve(ctx)
        retrieval_latency = time.time() - t0

        # 2. Media.
        t0 = time.time()
        media = self.media_processor.process(ctx)
        media_latency = time.time() - t0

        # 3. Policy.
        policy_decision = self.policy_engine.evaluate(ctx, retrieval, media)

        # 4. Router (LLM only if policy does not override).
        t0 = time.time()
        trace = self.router.route(ctx, retrieval=retrieval, media=media)
        llm_latency = time.time() - t0

        total_latency = time.time() - start

        # Record statistics.
        self.stats.record_success(
            retrieval_latency=retrieval_latency,
            media_latency=media_latency,
            llm_latency=llm_latency,
            total_latency=total_latency,
            used_policy=trace.llm_skipped,
        )

        # Build the trace dict.
        trace_dict = {
            "message_id": ctx.message_id,
            "retrieval_summary": retrieval.retrieval_summary,
            "media_summary": media.summary if media else "",
            "policy_used": trace.llm_skipped,
            "policy_rule": trace.policy_rule,
            "prompt_summary": trace.prompt_summary,
            "llm_response": trace.llm_response.to_dict(),
            "confidence_before": trace.validated.confidence,
            "confidence_after": trace.final_decision.confidence,
            "final_decision": trace.final_decision.to_dict(),
            "output_row": trace.final_decision.to_output_row(),
            "latencies": {
                "retrieval": round(retrieval_latency, 4),
                "media": round(media_latency, 4),
                "llm": round(llm_latency, 4),
                "total": round(total_latency, 4),
            },
        }

        return trace.final_decision.to_output_row(), trace_dict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_checkpoint_rows(self) -> list[dict]:
        """Load existing rows from the checkpoint file."""
        rows: list[dict] = []
        if not self.checkpoint.has_checkpoint():
            return rows
        try:
            import csv

            with open(self.checkpoint.partial_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(row) for row in reader]
        except Exception as exc:
            logger.warning("Failed to load checkpoint rows: %s", exc)
        return rows