"""Metrics computation for the evaluation framework.

Computes accuracy, precision, recall, F1 (macro, weighted, micro) for
both action and message_type predictions.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def compute_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict[str, float]:
    """Compute classification metrics for a single label set.

    Parameters
    ----------
    y_true : list[str]
        Ground truth labels.
    y_pred : list[str]
        Predicted labels.

    Returns
    -------
    dict[str, float]
        accuracy, precision (macro), recall (macro), f1 (macro),
        f1 (weighted), f1 (micro).
    """
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "f1_micro": 0.0, "per_label": {}}

    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)

    # Accuracy.
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # Per-label precision / recall / F1.
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    # Macro averages.
    macro_precision = sum(v["precision"] for v in per_label.values()) / len(labels)
    macro_recall = sum(v["recall"] for v in per_label.values()) / len(labels)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels)

    # Weighted F1 (by support).
    total_support = sum(v["support"] for v in per_label.values())
    weighted_f1 = (
        sum(v["f1"] * v["support"] for v in per_label.values()) / total_support
        if total_support > 0
        else 0.0
    )

    # Micro F1 = accuracy for single-label classification.
    micro_f1 = accuracy

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "f1_macro": round(macro_f1, 4),
        "f1_weighted": round(weighted_f1, 4),
        "f1_micro": round(micro_f1, 4),
        "per_label": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_label.items()},
    }


def compute_all_metrics(
    gt: pd.DataFrame, pred: pd.DataFrame
) -> dict[str, Any]:
    """Compute all metrics for action and message_type.

    Parameters
    ----------
    gt : pd.DataFrame
        Ground truth with columns: message_id, action, message_type.
    pred : pd.DataFrame
        Predictions with columns: message_id, action, message_type.

    Returns
    -------
    dict[str, Any]
        All metrics.
    """
    merged = gt.merge(pred, on="message_id", suffixes=("_true", "_pred"))

    action_metrics = compute_metrics(
        merged["action_true"].tolist(),
        merged["action_pred"].tolist(),
    )
    type_metrics = compute_metrics(
        merged["message_type_true"].tolist(),
        merged["message_type_pred"].tolist(),
    )

    return {
        "total_messages": len(merged),
        "action_accuracy": action_metrics["accuracy"],
        "message_type_accuracy": type_metrics["accuracy"],
        "action_precision": action_metrics["precision"],
        "action_recall": action_metrics["recall"],
        "action_f1_macro": action_metrics["f1_macro"],
        "action_f1_weighted": action_metrics["f1_weighted"],
        "action_f1_micro": action_metrics["f1_micro"],
        "message_type_precision": type_metrics["precision"],
        "message_type_recall": type_metrics["recall"],
        "message_type_f1_macro": type_metrics["f1_macro"],
        "message_type_f1_weighted": type_metrics["f1_weighted"],
        "message_type_f1_micro": type_metrics["f1_micro"],
        "overall_accuracy": round(
            (action_metrics["accuracy"] + type_metrics["accuracy"]) / 2, 4
        ),
        "action_per_label": action_metrics["per_label"],
        "message_type_per_label": type_metrics["per_label"],
    }