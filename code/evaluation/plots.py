"""Plot generation for the evaluation framework using matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(cm: pd.DataFrame, title: str, output_path: Path) -> None:
    """Save a confusion matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Annotate cells.
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            value = cm.iloc[i, j]
            color = "white" if value > cm.values.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confidence_histogram(confidences: list[float], output_path: Path) -> None:
    """Save a confidence histogram as PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=20, edgecolor="black", color="steelblue", alpha=0.7)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.axvline(0.5, color="red", linestyle="--", label="0.5 threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_action_distribution(pred: pd.DataFrame, output_path: Path) -> None:
    """Save action distribution bar chart as PNG."""
    counts = pred["action"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"notify": "#2196F3", "digest": "#4CAF50", "mute": "#F44336"}
    bars = ax.bar(counts.index, counts.values, color=[colors.get(a, "gray") for a in counts.index])
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title("Action Distribution")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(bar.get_height())), ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_message_type_distribution(pred: pd.DataFrame, output_path: Path) -> None:
    """Save message type distribution bar chart as PNG."""
    counts = pred["message_type"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(counts.index, counts.values, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Message Type Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latency_distribution(latencies: list[dict], output_path: Path) -> None:
    """Save latency distribution as PNG."""
    if not latencies:
        return
    df = pd.DataFrame(latencies)
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["retrieval", "media", "llm", "total"]
    data = [df[s].values for s in stages if s in df.columns]
    ax.boxplot(data, labels=stages[:len(data)])
    ax.set_ylabel("Seconds")
    ax.set_title("Latency Distribution by Stage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_precision_recall(metrics: dict[str, Any], output_path: Path) -> None:
    """Save precision-recall bar chart as PNG."""
    per_label = metrics.get("action_per_label", {})
    if not per_label:
        return

    labels = list(per_label.keys())
    precision = [per_label[l].get("precision", 0) for l in labels]
    recall = [per_label[l].get("recall", 0) for l in labels]
    f1 = [per_label[l].get("f1", 0) for l in labels]

    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width for i in x], precision, width, label="Precision", color="#2196F3")
    ax.bar(list(x), recall, width, label="Recall", color="#4CAF50")
    ax.bar([i + width for i in x], f1, width, label="F1", color="#FF9800")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 by Action")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_all_plots(
    pred: pd.DataFrame,
    confusion_matrices: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    confidence_analysis: dict[str, Any],
    latency_analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate all evaluation plots."""
    # Confusion matrix.
    if "action" in confusion_matrices:
        plot_confusion_matrix(
            confusion_matrices["action"],
            "Action Confusion Matrix",
            output_dir / "confusion_matrix.png",
        )

    # Confidence histogram.
    confidences = pd.to_numeric(pred["confidence"], errors="coerce").dropna().tolist()
    if confidences:
        plot_confidence_histogram(confidences, output_dir / "confidence_histogram.png")

    # Action distribution.
    plot_action_distribution(pred, output_dir / "action_distribution.png")

    # Message type distribution.
    plot_message_type_distribution(pred, output_dir / "message_type_distribution.png")

    # Latency distribution.
    if latency_analysis and "slowest_10" in latency_analysis:
        # Reconstruct from traces dir if available.
        pass  # Latency plot handled by evaluator with trace data

    # Precision-recall chart.
    plot_precision_recall(metrics, output_dir / "precision_recall_chart.png")