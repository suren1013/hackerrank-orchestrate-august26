"""Evaluator: orchestrates the full evaluation suite.

Compares predictions (output/output.csv) against ground truth
(dataset/sample_messages.csv or a provided ground_truth.csv) and
generates all evaluation artifacts in output/evaluation/.

This module is **read-only** — it never modifies output.csv or the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATASET_DIR, OUTPUT_DIR
from evaluation.analyzer import (
    build_misclassified,
    compute_confidence_analysis,
    compute_distributions,
    compute_latency_analysis,
)
from evaluation.confusion import save_confusion_matrices
from evaluation.metrics import compute_all_metrics
from evaluation.plots import generate_all_plots
from evaluation.report import (
    generate_html_report,
    generate_markdown_report,
    generate_prompt_analysis,
)
from evaluation.thresholds import generate_threshold_recommendations
from utils.logger import get_logger

logger = get_logger(__name__)

EVAL_DIRNAME = "evaluation"


class Evaluator:
    """Runs the full evaluation suite against ground truth."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        dataset_dir: Path | str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir or OUTPUT_DIR).resolve()
        self.dataset_dir = Path(dataset_dir or DATASET_DIR).resolve()
        self.eval_dir = self.output_dir / EVAL_DIRNAME
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir = self.output_dir / "traces"

    def evaluate(self, ground_truth_path: Path | str | None = None) -> dict[str, Any]:
        """Run the full evaluation and return all results.

        Parameters
        ----------
        ground_truth_path : Path | str | None
            Path to ground truth CSV. If None, uses dataset/sample_messages.csv.

        Returns
        -------
        dict[str, Any]
            All evaluation results.
        """
        # 1. Load ground truth and predictions.
        gt_path = Path(ground_truth_path) if ground_truth_path else (
            self.dataset_dir / "sample_messages.csv"
        )
        if not gt_path.exists():
            logger.error("Ground truth not found: %s", gt_path)
            return {}

        pred_path = self.output_dir / "output.csv"
        if not pred_path.exists():
            logger.error("Predictions not found: %s", pred_path)
            return {}

        gt_df = pd.read_csv(gt_path)
        pred_df = pd.read_csv(pred_path)

        # Ensure required columns.
        gt_cols = ["message_id", "action", "message_type"]
        pred_cols = ["message_id", "action", "message_type", "confidence", "reason"]

        gt_df = gt_df[[c for c in gt_cols if c in gt_df.columns]]
        pred_df = pred_df[[c for c in pred_cols if c in pred_df.columns]]

        logger.info("Evaluating %d predictions against %d ground truth labels.",
                     len(pred_df), len(gt_df))

        # 2. Compute metrics.
        metrics = compute_all_metrics(gt_df, pred_df)
        self._save_json(metrics, "metrics.json")
        logger.info("Action accuracy: %.1f%%", metrics.get("action_accuracy", 0) * 100)

        # 3. Confusion matrices.
        confusion_matrices = save_confusion_matrices(gt_df, pred_df, self.eval_dir)

        # 4. Distributions.
        contexts_df = pd.read_csv(self.dataset_dir / "messages.csv") if (
            self.dataset_dir / "messages.csv"
        ).exists() else None
        distributions = compute_distributions(pred_df, contexts_df)
        self._save_json(distributions, "distribution.json")

        # 5. Misclassified.
        misclassified = build_misclassified(gt_df, pred_df, self.traces_dir)
        if not misclassified.empty:
            misclassified.to_csv(self.eval_dir / "misclassified.csv", index=False)
            logger.info("Misclassified: %d messages", len(misclassified))

        # 6. Confidence analysis.
        confidence_analysis = compute_confidence_analysis(gt_df, pred_df)
        self._save_json(confidence_analysis, "confidence_analysis.json")

        # Confidence distribution CSV.
        if "confidence" in pred_df.columns:
            conf_df = pred_df[["message_id", "confidence"]].copy()
            conf_df.to_csv(self.eval_dir / "confidence_distribution.csv", index=False)

        # 7. Latency analysis.
        latency_analysis = compute_latency_analysis(self.traces_dir)
        if latency_analysis:
            self._save_json(latency_analysis, "latency_analysis.json")

        # 8. Threshold recommendations.
        threshold_report = generate_threshold_recommendations(
            metrics, confidence_analysis, misclassified
        )
        (self.eval_dir / "threshold_recommendations.md").write_text(
            threshold_report, encoding="utf-8"
        )

        # 9. Prompt analysis.
        prompt_analysis = generate_prompt_analysis(misclassified)
        (self.eval_dir / "prompt_analysis.md").write_text(
            prompt_analysis, encoding="utf-8"
        )

        # 10. Plots.
        try:
            generate_all_plots(
                pred_df, confusion_matrices, metrics,
                confidence_analysis, latency_analysis, self.eval_dir,
            )
            logger.info("Plots generated in %s", self.eval_dir)
        except Exception as exc:
            logger.warning("Plot generation failed: %s", exc)

        # 11. Reports.
        markdown_report = generate_markdown_report(
            metrics, confusion_matrices, distributions,
            confidence_analysis, latency_analysis,
            misclassified, threshold_report, prompt_analysis,
        )
        (self.eval_dir / "evaluation_report.md").write_text(
            markdown_report, encoding="utf-8"
        )

        html_report = generate_html_report(markdown_report)
        (self.eval_dir / "evaluation_report.html").write_text(
            html_report, encoding="utf-8"
        )

        logger.info("Evaluation report saved to %s", self.eval_dir)

        # Print summary.
        self._print_summary(metrics, confidence_analysis, misclassified)

        return {
            "metrics": metrics,
            "distributions": distributions,
            "confidence_analysis": confidence_analysis,
            "latency_analysis": latency_analysis,
            "misclassified_count": len(misclassified),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_json(self, data: dict[str, Any], filename: str) -> None:
        """Save a dict as JSON in the evaluation directory."""
        path = self.eval_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _print_summary(
        self,
        metrics: dict[str, Any],
        confidence: dict[str, Any],
        misclassified: pd.DataFrame,
    ) -> None:
        """Print a concise evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Messages     : {metrics.get('total_messages', 0)}")
        print(f"Action Accuracy    : {metrics.get('action_accuracy', 0):.1%}")
        print(f"Type Accuracy      : {metrics.get('message_type_accuracy', 0):.1%}")
        print(f"Overall Accuracy   : {metrics.get('overall_accuracy', 0):.1%}")
        print(f"Action F1 (macro)  : {metrics.get('action_f1_macro', 0):.4f}")
        print(f"Action F1 (weighted): {metrics.get('action_f1_weighted', 0):.4f}")
        print("-" * 60)
        print(f"Avg Confidence     : {confidence.get('average_confidence', 0):.4f}")
        print(f"Calibration Gap    : {confidence.get('calibration_gap', 0):.4f}")
        print(f"Misclassified      : {len(misclassified)}")
        print("=" * 60)
        print(f"Reports saved to: {self.eval_dir}")
        print()