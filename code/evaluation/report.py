"""HTML and Markdown report generation for the evaluation framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_markdown_report(
    metrics: dict[str, Any],
    confusion_matrices: dict[str, pd.DataFrame],
    distributions: dict[str, Any],
    confidence_analysis: dict[str, Any],
    latency_analysis: dict[str, Any],
    misclassified: pd.DataFrame,
    threshold_report: str,
    prompt_analysis: str,
) -> str:
    """Generate a comprehensive markdown evaluation report."""
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total Messages | {metrics.get('total_messages', 0)} |")
    lines.append(f"| Action Accuracy | {metrics.get('action_accuracy', 0):.1%} |")
    lines.append(f"| Message Type Accuracy | {metrics.get('message_type_accuracy', 0):.1%} |")
    lines.append(f"| Overall Accuracy | {metrics.get('overall_accuracy', 0):.1%} |")
    lines.append(f"| Action Precision (macro) | {metrics.get('action_precision', 0):.4f} |")
    lines.append(f"| Action Recall (macro) | {metrics.get('action_recall', 0):.4f} |")
    lines.append(f"| Action F1 (macro) | {metrics.get('action_f1_macro', 0):.4f} |")
    lines.append(f"| Action F1 (weighted) | {metrics.get('action_f1_weighted', 0):.4f} |")
    lines.append(f"| Action F1 (micro) | {metrics.get('action_f1_micro', 0):.4f} |")
    lines.append(f"| Type F1 (macro) | {metrics.get('message_type_f1_macro', 0):.4f} |")
    lines.append(f"| Type F1 (weighted) | {metrics.get('message_type_f1_weighted', 0):.4f} |")
    lines.append("")

    # Confusion matrix.
    if "action" in confusion_matrices:
        cm = confusion_matrices["action"]
        lines.append("## Action Confusion Matrix")
        lines.append("")
        lines.append("| Actual \\ Predicted | " + " | ".join(cm.columns) + " |")
        lines.append("|---|" + "---|" * len(cm.columns))
        for idx in cm.index:
            row_vals = " | ".join(str(cm.loc[idx, c]) for c in cm.columns)
            lines.append(f"| {idx} | {row_vals} |")
        lines.append("")

    # Distributions.
    lines.append("## Distributions")
    lines.append("")
    if "action" in distributions:
        lines.append("### Action Distribution")
        lines.append("| Action | Count | Percent |")
        lines.append("|---|---|---|")
        for a, info in distributions["action"].items():
            lines.append(f"| {a} | {info['count']} | {info['percent']}% |")
        lines.append("")

    # Confidence analysis.
    lines.append("## Confidence Analysis")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Average Confidence | {confidence_analysis.get('average_confidence', 0):.4f} |")
    lines.append(f"| Median Confidence | {confidence_analysis.get('median_confidence', 0):.4f} |")
    lines.append(f"| Avg Confidence (correct) | {confidence_analysis.get('avg_confidence_correct', 0):.4f} |")
    lines.append(f"| Avg Confidence (incorrect) | {confidence_analysis.get('avg_confidence_incorrect', 0):.4f} |")
    lines.append(f"| Calibration Gap | {confidence_analysis.get('calibration_gap', 0):.4f} |")
    lines.append("")

    # Latency analysis.
    if latency_analysis:
        lines.append("## Latency Analysis")
        lines.append("")
        lines.append(f"| Stage | Average (s) |")
        lines.append(f"|---|---|")
        lines.append(f"| Retrieval | {latency_analysis.get('avg_retrieval', 0):.4f} |")
        lines.append(f"| Media | {latency_analysis.get('avg_media', 0):.4f} |")
        lines.append(f"| LLM | {latency_analysis.get('avg_llm', 0):.4f} |")
        lines.append(f"| Total | {latency_analysis.get('avg_total', 0):.4f} |")
        lines.append("")

    # Top 20 failures.
    lines.append("## Top 20 Failures (by confidence)")
    lines.append("")
    if not misclassified.empty:
        lines.append("| message_id | expected | predicted | confidence | reason |")
        lines.append("|---|---|---|---|---|")
        for _, row in misclassified.head(20).iterrows():
            reason = str(row.get("reason", ""))[:60]
            lines.append(
                f"| {row['message_id']} | {row['expected_action']}/{row['expected_type']} | "
                f"{row['predicted_action']}/{row['predicted_type']} | "
                f"{row.get('confidence', 0):.4f} | {reason} |"
            )
    else:
        lines.append("No misclassified messages found!")
    lines.append("")

    # Best/worst categories.
    lines.append("## Best/Worst Performing Categories")
    lines.append("")
    per_label = metrics.get("action_per_label", {})
    if per_label:
        sorted_labels = sorted(per_label.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
        lines.append("| Action | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for label, info in sorted_labels:
            lines.append(
                f"| {label} | {info.get('precision', 0):.4f} | {info.get('recall', 0):.4f} | "
                f"{info.get('f1', 0):.4f} | {info.get('support', 0)} |"
            )
    lines.append("")

    # Threshold recommendations.
    lines.append("## Threshold Recommendations")
    lines.append("")
    lines.append(threshold_report)
    lines.append("")

    # Prompt analysis.
    lines.append("## Prompt Analysis")
    lines.append("")
    lines.append(prompt_analysis)
    lines.append("")

    return "\n".join(lines)


def generate_html_report(markdown: str) -> str:
    """Convert a markdown report to a simple HTML page."""
    # Simple markdown-to-HTML conversion.
    html_lines: list[str] = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html><head><meta charset='utf-8'>")
    html_lines.append("<title>Evaluation Report</title>")
    html_lines.append("<style>")
    html_lines.append("body { font-family: Arial, sans-serif; margin: 40px; max-width: 1000px; }")
    html_lines.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
    html_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html_lines.append("th { background-color: #f4f4f4; }")
    html_lines.append("h1, h2, h3 { color: #333; }")
    html_lines.append("img { max-width: 100%; margin: 10px 0; }")
    html_lines.append("</style></head><body>")

    in_table = False
    for line in markdown.split("\n"):
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells and all(set(c) <= set("-: ") for c in cells):
                continue  # Skip separator row
            tag = "th" if not html_lines[-1].startswith("<tr") else "td"
            if not html_lines[-1].startswith("<tr"):
                html_lines.append("<tr>")
                tag = "th"
            html_lines.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            if line.strip():
                html_lines.append(f"<p>{line}</p>")

    if in_table:
        html_lines.append("</table>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def generate_prompt_analysis(misclassified: pd.DataFrame) -> str:
    """Group failures by reason and recommend prompt improvements."""
    lines: list[str] = []
    lines.append("### Failure Patterns")
    lines.append("")

    if misclassified.empty:
        lines.append("No misclassified messages — no prompt improvements needed.")
        return "\n".join(lines)

    # Group by expected vs predicted action.
    patterns = misclassified.groupby(["expected_action", "predicted_action"]).size().reset_index(name="count")
    patterns = patterns.sort_values("count", ascending=False)

    lines.append("| Expected | Predicted | Count |")
    lines.append("|---|---|---|")
    for _, row in patterns.iterrows():
        lines.append(f"| {row['expected_action']} | {row['predicted_action']} | {row['count']} |")
    lines.append("")

    # Recommendations based on patterns.
    lines.append("### Recommendations")
    lines.append("")

    for _, row in patterns.iterrows():
        exp = row["expected_action"]
        pred = row["predicted_action"]
        count = row["count"]

        if exp == "notify" and pred == "digest":
            lines.append(f"- **{count}x notify→digest**: Messages that should interrupt were digested. Consider strengthening urgency detection in the prompt.")
        elif exp == "notify" and pred == "mute":
            lines.append(f"- **{count}x notify→mute**: Important messages were muted. Review policy scam threshold — it may be too aggressive.")
        elif exp == "digest" and pred == "notify":
            lines.append(f"- **{count}x digest→notify**: Non-urgent messages were notified. Add stricter urgency criteria to the prompt.")
        elif exp == "digest" and pred == "mute":
            lines.append(f"- **{count}x digest→mute**: Useful messages were muted. Review policy rules for over-muting.")
        elif exp == "mute" and pred == "notify":
            lines.append(f"- **{count}x mute→notify**: Scams/spam were notified! This is critical — strengthen safety detection.")
        elif exp == "mute" and pred == "digest":
            lines.append(f"- **{count}x mute→digest**: Scams/spam were digested instead of muted. Lower the scam detection threshold.")
        else:
            lines.append(f"- **{count}x {exp}→{pred}**: Review this pattern in the traces.")

    lines.append("")
    return "\n".join(lines)