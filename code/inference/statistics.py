"""Statistics collector for the inference pipeline.

Tracks per-stage latencies, success/failure counts, policy vs LLM decisions,
and produces a clean formatted report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineStatistics:
    """Collects and reports inference pipeline statistics."""

    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    messages_processed: int = 0
    messages_succeeded: int = 0
    messages_failed: int = 0
    policy_decisions: int = 0
    llm_decisions: int = 0

    retrieval_latencies: list[float] = field(default_factory=list)
    media_latencies: list[float] = field(default_factory=list)
    llm_latencies: list[float] = field(default_factory=list)
    total_latencies: list[float] = field(default_factory=list)

    failed_messages: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_success(
        self,
        retrieval_latency: float,
        media_latency: float,
        llm_latency: float,
        total_latency: float,
        used_policy: bool,
    ) -> None:
        """Record a successful message processing."""
        self.messages_processed += 1
        self.messages_succeeded += 1
        self.retrieval_latencies.append(retrieval_latency)
        self.media_latencies.append(media_latency)
        self.llm_latencies.append(llm_latency)
        self.total_latencies.append(total_latency)
        if used_policy:
            self.policy_decisions += 1
        else:
            self.llm_decisions += 1

    def record_failure(self, message_id: str) -> None:
        """Record a failed message."""
        self.messages_processed += 1
        self.messages_failed += 1
        self.failed_messages.append(message_id)

    def finish(self) -> None:
        """Mark the pipeline as finished."""
        self.end_time = time.time()

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    @property
    def total_runtime(self) -> float:
        """Total runtime in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0..1)."""
        if self.messages_processed == 0:
            return 0.0
        return self.messages_succeeded / self.messages_processed

    def _avg(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return statistics as a dict."""
        return {
            "messages_processed": self.messages_processed,
            "messages_succeeded": self.messages_succeeded,
            "messages_failed": self.messages_failed,
            "policy_decisions": self.policy_decisions,
            "llm_decisions": self.llm_decisions,
            "avg_retrieval_latency": round(self._avg(self.retrieval_latencies), 4),
            "avg_media_latency": round(self._avg(self.media_latencies), 4),
            "avg_llm_latency": round(self._avg(self.llm_latencies), 4),
            "avg_total_latency": round(self._avg(self.total_latencies), 4),
            "total_runtime": round(self.total_runtime, 4),
            "success_rate": round(self.success_rate, 4),
            "failed_messages": self.failed_messages,
        }

    def print_report(self) -> None:
        """Print a clean formatted statistics report."""
        print("\n" + "=" * 60)
        print("INFERENCE PIPELINE REPORT")
        print("=" * 60)
        print(f"Messages processed : {self.messages_processed}")
        print(f"Messages succeeded : {self.messages_succeeded}")
        print(f"Messages failed    : {self.messages_failed}")
        print(f"Policy decisions   : {self.policy_decisions}")
        print(f"LLM decisions      : {self.llm_decisions}")
        print(f"Success rate       : {self.success_rate:.1%}")
        print("-" * 60)
        print(f"Avg retrieval      : {self._avg(self.retrieval_latencies):.3f}s")
        print(f"Avg media          : {self._avg(self.media_latencies):.3f}s")
        print(f"Avg LLM            : {self._avg(self.llm_latencies):.3f}s")
        print(f"Avg total          : {self._avg(self.total_latencies):.3f}s")
        print(f"Total runtime      : {self.total_runtime:.3f}s")
        print("=" * 60 + "\n")

        if self.failed_messages:
            print("Failed Messages:")
            for mid in self.failed_messages:
                print(f"  {mid}")
            print()