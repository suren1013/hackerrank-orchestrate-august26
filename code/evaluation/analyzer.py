"""Distribution, error, confidence, and latency analysis."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from utils.helpers import safe_str


def compute_distributions(pred: pd.DataFrame, contexts_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute action, conversation, and media distributions."""
    n = len(pred)
    if n == 0:
        return {}

    dist: dict[str, Any] = {}

    # Action distribution.
    action_counts = pred["action"].value_counts()
    dist["action"] = {
        a: {"count": int(action_counts.get(a, 0)), "percent": round(action_counts.get(a, 0) / n * 100, 2)}
        for a in ["notify", "digest", "mute"]
    }

    # Conversation type distribution (from contexts if available).
    if contexts_df is not None and "conversation_type" in contexts_df.columns:
        conv_counts = contexts_df["conversation_type"].value_counts()
        dist["conversation_type"] = {
            c: {"count": int(conv_counts.get(c, 0)), "percent": round(conv_counts.get(c, 0) / n * 100, 2)}
            for c in ["business", "personal", "group"]
        }

    # Media distribution.
    if contexts_df is not None and "media_type" in contexts_df.columns:
        media_counts = contexts_df["media_type"].fillna("").apply(lambda x: x if x else "text").value_counts()
        dist["media_type"] = {
            m: {"count": int(media_counts.get(m, 0)), "percent": round(media_counts.get(m, 0) / n * 100, 2)}
            for m in ["text", "image", "voice"]
        }

    # Message type distribution.
    type_counts = pred["message_type"].value_counts()
    dist["message_type"] = {
        t: {"count": int(type_counts.get(t, 0)), "percent": round(type_counts.get(t, 0) / n * 100, 2)}
        for t in type_counts.index
    }

    return dist


def build_misclassified(
    gt: pd.DataFrame, pred: pd.DataFrame, traces_dir: Path | None = None
) -> pd.DataFrame:
    """Build a DataFrame of misclassified messages sorted by confidence.

    Enriches with trace data (policy_used, policy_rule, retrieval_summary)
    if traces are available.
    """
    merged = gt.merge(pred, on="message_id", suffixes=("_true", "_pred"))

    # Filter to misclassified (action or type wrong).
    misclassified = merged[
        (merged["action_true"] != merged["action_pred"])
        | (merged["message_type_true"] != merged["message_type_pred"])
    ].copy()

    if misclassified.empty:
        return pd.DataFrame(columns=[
            "message_id", "expected_action", "predicted_action",
            "expected_type", "predicted_type", "confidence", "reason",
            "policy_used", "policy_rule", "llm_used", "top_evidence",
            "retrieval_summary",
        ])

    # Rename columns.
    misclassified = misclassified.rename(columns={
        "action_true": "expected_action",
        "action_pred": "predicted_action",
        "message_type_true": "expected_type",
        "message_type_pred": "predicted_type",
    })

    # Enrich with trace data.
    if traces_dir is not None and traces_dir.exists():
        for idx, row in misclassified.iterrows():
            mid = row["message_id"]
            trace_path = traces_dir / f"{mid}.json"
            if trace_path.exists():
                try:
                    trace = json.loads(trace_path.read_text(encoding="utf-8"))
                    misclassified.at[idx, "policy_used"] = trace.get("policy_used", False)
                    misclassified.at[idx, "policy_rule"] = trace.get("policy_rule") or ""
                    misclassified.at[idx, "llm_used"] = not trace.get("policy_used", False)
                    misclassified.at[idx, "top_evidence"] = (
                        trace.get("final_decision", {}).get("evidence_message_ids", [""])[0]
                        if trace.get("final_decision", {}).get("evidence_message_ids")
                        else ""
                    )
                    misclassified.at[idx, "retrieval_summary"] = trace.get("retrieval_summary", "")
                except Exception:
                    pass

    # Sort by confidence descending (highest confidence mistakes first).
    if "confidence" in misclassified.columns:
        misclassified["confidence"] = pd.to_numeric(misclassified["confidence"], errors="coerce")
        misclassified = misclassified.sort_values("confidence", ascending=False)

    # Select and order columns.
    columns = [
        "message_id", "expected_action", "predicted_action",
        "expected_type", "predicted_type", "confidence", "reason",
        "policy_used", "policy_rule", "llm_used", "top_evidence",
        "retrieval_summary",
    ]
    for col in columns:
        if col not in misclassified.columns:
            misclassified[col] = ""

    return misclassified[columns].reset_index(drop=True)


def compute_confidence_analysis(
    gt: pd.DataFrame, pred: pd.DataFrame
) -> dict[str, Any]:
    """Compute confidence statistics and calibration gap."""
    merged = gt.merge(pred, on="message_id", suffixes=("_true", "_pred"))
    if merged.empty or "confidence" not in merged.columns:
        return {}

    confidences = pd.to_numeric(merged["confidence"], errors="coerce").dropna()
    if confidences.empty:
        return {}

    # Correct vs incorrect.
    correct_mask = (merged["action_true"] == merged["action_pred"])
    correct_conf = pd.to_numeric(merged.loc[correct_mask, "confidence"], errors="coerce").dropna()
    incorrect_conf = pd.to_numeric(merged.loc[~correct_mask, "confidence"], errors="coerce").dropna()

    avg_correct = float(correct_conf.mean()) if not correct_conf.empty else 0.0
    avg_incorrect = float(incorrect_conf.mean()) if not incorrect_conf.empty else 0.0

    return {
        "average_confidence": round(float(confidences.mean()), 4),
        "median_confidence": round(float(confidences.median()), 4),
        "lowest_confidence": round(float(confidences.min()), 4),
        "highest_confidence": round(float(confidences.max()), 4),
        "avg_confidence_correct": round(avg_correct, 4),
        "avg_confidence_incorrect": round(avg_incorrect, 4),
        "calibration_gap": round(avg_correct - avg_incorrect, 4),
        "total_correct": int(len(correct_conf)),
        "total_incorrect": int(len(incorrect_conf)),
    }


def compute_latency_analysis(traces_dir: Path) -> dict[str, Any]:
    """Analyze latencies from saved trace files."""
    if not traces_dir.exists():
        return {}

    latencies: list[dict[str, float]] = []
    for trace_file in traces_dir.glob("*.json"):
        try:
            trace = json.loads(trace_file.read_text(encoding="utf-8"))
            lat = trace.get("latencies", {})
            latencies.append({
                "message_id": trace.get("message_id", ""),
                "retrieval": lat.get("retrieval", 0),
                "media": lat.get("media", 0),
                "llm": lat.get("llm", 0),
                "total": lat.get("total", 0),
            })
        except Exception:
            continue

    if not latencies:
        return {}

    df = pd.DataFrame(latencies)

    # Slowest and fastest 10.
    slowest = df.nlargest(10, "total")[["message_id", "total"]].to_dict("records")
    fastest = df.nsmallest(10, "total")[["message_id", "total"]].to_dict("records")

    return {
        "avg_retrieval": round(float(df["retrieval"].mean()), 4),
        "avg_media": round(float(df["media"].mean()), 4),
        "avg_llm": round(float(df["llm"].mean()), 4),
        "avg_total": round(float(df["total"].mean()), 4),
        "slowest_10": slowest,
        "fastest_10": fastest,
    }