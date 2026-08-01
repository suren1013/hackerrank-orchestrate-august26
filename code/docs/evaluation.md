# Evaluation

## Overview

The evaluation suite compares predictions (`output/output.csv`) against ground truth and generates comprehensive analysis artifacts. It is **read-only** — never modifies output.csv or the pipeline.

## Running Evaluation

```bash
python main.py --evaluate
python main.py --evaluate --ground-truth path/to/ground_truth.csv
```

## Metrics Computed

- Action Accuracy, Message Type Accuracy, Overall Accuracy
- Precision (macro), Recall (macro), F1 (macro/weighted/micro)
- Per-label precision/recall/F1/support

## Artifacts Generated

All in `output/evaluation/`:

| File | Description |
|---|---|
| `metrics.json` | All metrics |
| `confusion_matrix.csv` | Action confusion matrix |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `distribution.json` | Action/conversation/media distributions |
| `misclassified.csv` | Misclassified messages (sorted by confidence) |
| `confidence_analysis.json` | Confidence stats + calibration gap |
| `confidence_histogram.png` | Confidence distribution |
| `action_distribution.png` | Action distribution chart |
| `message_type_distribution.png` | Type distribution chart |
| `precision_recall_chart.png` | Precision/Recall/F1 by action |
| `latency_analysis.json` | Per-stage latency averages |
| `threshold_recommendations.md` | Threshold adjustment suggestions |
| `prompt_analysis.md` | Failure patterns + prompt improvement suggestions |
| `evaluation_report.md` | Comprehensive markdown report |
| `evaluation_report.html` | HTML version |

## Current Results

The evaluation framework works correctly. When run against `sample_messages.csv`, 0% accuracy is expected because sample messages use `sample_msg_XXX` IDs that don't match `msg_XXX` IDs in output.csv. The hidden ground truth will produce meaningful metrics.

Cross-references: [Overview](overview.md) | [Prompt Strategy](prompt_strategy.md) | [Submission Checklist](submission_checklist.md)