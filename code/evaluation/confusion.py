"""Confusion matrix generation for the evaluation framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> pd.DataFrame:
    """Build a confusion matrix as a DataFrame.

    Parameters
    ----------
    y_true : list[str]
        Ground truth labels.
    y_pred : list[str]
        Predicted labels.
    labels : list[str] | None
        Ordered list of labels for rows/columns.

    Returns
    -------
    pd.DataFrame
        Confusion matrix with true labels as rows, predicted as columns.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    matrix = {label: {l: 0 for l in labels} for label in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    df = pd.DataFrame(matrix).T
    df.index.name = "actual"
    return df


def save_confusion_matrices(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build and save action and message_type confusion matrices.

    Returns
    -------
    dict[str, pd.DataFrame]
        {"action": df, "message_type": df}
    """
    merged = gt.merge(pred, on="message_id", suffixes=("_true", "_pred"))

    action_cm = build_confusion_matrix(
        merged["action_true"].tolist(),
        merged["action_pred"].tolist(),
        labels=["notify", "digest", "mute"],
    )
    type_labels = sorted(
        set(merged["message_type_true"]) | set(merged["message_type_pred"])
    )
    type_cm = build_confusion_matrix(
        merged["message_type_true"].tolist(),
        merged["message_type_pred"].tolist(),
        labels=type_labels,
    )

    # Save as CSV.
    action_cm.to_csv(output_dir / "confusion_matrix_action.csv")
    type_cm.to_csv(output_dir / "confusion_matrix_type.csv")

    # Combined CSV.
    combined = pd.DataFrame(
        {"actual": action_cm.index.tolist()},
    )
    action_cm.to_csv(output_dir / "confusion_matrix.csv")

    return {"action": action_cm, "message_type": type_cm}