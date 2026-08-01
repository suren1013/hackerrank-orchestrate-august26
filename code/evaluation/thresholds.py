"""Threshold analysis and recommendations.

Analyzes current thresholds and generates recommendations without
automatically changing them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_threshold_recommendations(
    metrics: dict[str, Any],
    confidence_analysis: dict[str, Any],
    misclassified: pd.DataFrame,
) -> str:
    """Generate a markdown report with threshold recommendations."""
    lines: list[str] = []
    lines.append("# Threshold Recommendations")
    lines.append("")
    lines.append("> These are **recommendations only**. No thresholds are changed automatically.")
    lines.append("")

    # Action accuracy.
    action_acc = metrics.get("action_accuracy", 0)
    lines.append(f"## Current Performance")
    lines.append(f"- Action Accuracy: {action_acc:.1%}")
    lines.append(f"- Message Type Accuracy: {metrics.get('message_type_accuracy', 0):.1%}")
    lines.append(f"- Overall Accuracy: {metrics.get('overall_accuracy', 0):.1%}")
    lines.append("")

    # Confidence calibration.
    cal_gap = confidence_analysis.get("calibration_gap", 0)
    lines.append("## Confidence Calibration")
    lines.append(f"- Avg confidence (correct): {confidence_analysis.get('avg_confidence_correct', 0):.4f}")
    lines.append(f"- Avg confidence (incorrect): {confidence_analysis.get('avg_confidence_incorrect', 0):.4f}")
    lines.append(f"- Calibration gap: {cal_gap:.4f}")
    if cal_gap < 0.05:
        lines.append("- **Assessment**: Well calibrated — correct predictions have higher confidence than incorrect ones.")
    elif cal_gap < 0.15:
        lines.append("- **Assessment**: Moderately calibrated — consider tightening confidence for borderline cases.")
    else:
        lines.append("- **Assessment**: Poorly calibrated — the model is overconfident on wrong predictions. Review confidence calibration weights.")
    lines.append("")

    # Threshold recommendations.
    lines.append("## Recommended Threshold Adjustments")
    lines.append("")

    # Policy scam threshold.
    scam_errors = misclassified[
        (misclassified.get("expected_action") == "mute") &
        (misclassified.get("predicted_action") != "mute")
    ] if not misclassified.empty else pd.DataFrame()
    lines.append("### Policy Scam Detection Threshold")
    lines.append(f"- Current: trust_score < 0.4 triggers scam mute")
    if len(scam_errors) > 0:
        lines.append(f"- Missed scams: {len(scam_errors)}")
        lines.append(f"- **Recommendation**: Lower threshold to 0.5 or add more safety keywords to catch missed scams.")
    else:
        lines.append("- **Recommendation**: Current threshold appears effective. No change needed.")
    lines.append("")

    # Notify threshold.
    notify_errors = misclassified[
        (misclassified.get("predicted_action") == "notify") &
        (misclassified.get("expected_action") != "notify")
    ] if not misclassified.empty else pd.DataFrame()
    lines.append("### Notify Threshold")
    lines.append(f"- Current: LLM decides based on urgency signals")
    if len(notify_errors) > 0:
        lines.append(f"- False notifies: {len(notify_errors)}")
        lines.append(f"- **Recommendation**: Add stricter urgency criteria to reduce false notifications.")
    else:
        lines.append("- **Recommendation**: No false notifications detected.")
    lines.append("")

    # Forward threshold.
    forward_errors = misclassified[
        (misclassified.get("expected_action") == "mute") &
        (misclassified.get("predicted_action") == "notify")
    ] if not misclassified.empty else pd.DataFrame()
    lines.append("### Forward/Spam Threshold")
    lines.append(f"- Current: forwarded_count >= 5 + ignore_rate >= 50%")
    if len(forward_errors) > 0:
        lines.append(f"- Forwards that were notified instead of muted: {len(forward_errors)}")
        lines.append(f"- **Recommendation**: Lower forward threshold to 3 or reduce ignore rate requirement.")
    else:
        lines.append("- **Recommendation**: Current threshold appears effective.")
    lines.append("")

    # Trust / Interest thresholds.
    lines.append("### Trust Score Threshold")
    lines.append(f"- Current: trust_score < 0.4 = low trust (scam risk)")
    lines.append("- **Recommendation**: Review per-message trust scores in traces to fine-tune.")
    lines.append("")

    lines.append("### Interest Score Threshold")
    lines.append(f"- Current: interest_score contributes to notify/digest decision")
    lines.append("- **Recommendation**: Analyze interest scores of misclassified messages to adjust weights.")
    lines.append("")

    return "\n".join(lines)