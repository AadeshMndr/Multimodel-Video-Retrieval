# Report Dashboard Guide

This repository now includes a Streamlit dashboard for exploring evaluation reports.

## What it shows

- Run-level summary metrics
- Per-path comparison for clip, xclip, yolo, audio, and ocr results when present
- Prompt-level results with filters for path and IoU
- Processing-time versus IoU scatter plots
- Hyperparameter comparison tables and threshold exploration
- Duration-overlap metrics including temporal set IoU, overlap over max duration, duration precision, and duration recall
- Per-video drilldown for timestamps, route details, and IoU values
- Raw JSON inspection for the report files

## Run it

From the project root:

```bash
streamlit run report_dashboard.py
```

If Streamlit is not installed in your environment, install the project requirements first.

## Data sources

The dashboard reads from:

- evaluation/reports/cumulative_summary.json
- evaluation/reports/cumulative_prompt_results.jsonl
- evaluation/reports/runs/*/run_summary.json
- evaluation/reports/runs/*/run_state.json
- evaluation/reports/runs/*/per_video/*.json

## Notes

- The dashboard is read-only and does not modify report files.
- It is designed for the existing evaluation outputs in this repository, especially the Jungle.mp4 run data currently present under evaluation/reports/runs.

## Field glossary

See [Report Summary Fields](report_summary_fields.md) for a detailed explanation of the summary metrics, IoU fields, Recall@IoU, and hyperparameter combo values.
