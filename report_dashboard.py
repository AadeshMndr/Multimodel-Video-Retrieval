from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "evaluation" / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
CUMULATIVE_SUMMARY_PATH = REPORTS_DIR / "cumulative_summary.json"
CUMULATIVE_RESULTS_PATH = REPORTS_DIR / "cumulative_prompt_results.jsonl"

PATH_ORDER = ["clip", "xclip", "yolo", "audio", "ocr"]
PATH_COLORS = {
    "clip": "#d96c75",
    "xclip": "#3f8cff",
    "yolo": "#2ab07f",
    "audio": "#f4a261",
    "ocr": "#9c6ade",
}


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_all_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True):
        summary_path = run_dir / "run_summary.json"
        state_path = run_dir / "run_state.json"
        summary = load_json(summary_path)
        state = load_json(state_path)
        if summary:
            record = {
                "run_dir": run_dir,
                "run_id": summary.get("run_id", run_dir.name),
                "status": summary.get("status", state.get("status", "unknown")),
                "started_utc": summary.get("started_utc", state.get("started_utc")),
                "finished_utc": summary.get("finished_utc", state.get("finished_utc")),
                "summary": summary,
                "state": state,
            }
            runs.append(record)
    return runs


def flatten_path_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.get("summary_by_path", []):
        path_taken = item.get("path_taken")
        path_summary = item.get("summary", {})
        rows.append(
            {
                "path_taken": path_taken,
                "prompt_count": path_summary.get("prompt_count", 0),
                "avg_best_iou": path_summary.get("avg_best_iou", 0.0),
                "avg_recall": path_summary.get("avg_recall", 0.0),
                "avg_overlap_anywhere_recall": path_summary.get("avg_overlap_anywhere_recall", 0.0),
                "avg_processing_seconds": path_summary.get("avg_processing_seconds", 0.0),
                "total_processing_seconds": path_summary.get("total_processing_seconds", 0.0),
                "total_predicted_total_duration_seconds": path_summary.get("total_predicted_total_duration_seconds", 0.0),
            }
        )
    return rows


def flatten_combo_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.get("summary_by_hyperparameter_combo", []):
        combo_summary = item.get("summary", {})
        hyperparameters = item.get("hyperparameters", {})
        rows.append(
            {
                "hyperparameter_combo_key": item.get("hyperparameter_combo_key"),
                "prompt_count": combo_summary.get("prompt_count", 0),
                "avg_best_iou": combo_summary.get("avg_best_iou", 0.0),
                "avg_recall": combo_summary.get("avg_recall", 0.0),
                "avg_overlap_anywhere_recall": combo_summary.get("avg_overlap_anywhere_recall", 0.0),
                "avg_processing_seconds": combo_summary.get("avg_processing_seconds", 0.0),
                "total_predicted_total_duration_seconds": combo_summary.get("total_predicted_total_duration_seconds", 0.0),
                "CLIP_THRESHOLD": hyperparameters.get("CLIP_THRESHOLD"),
                "XCLIP_THRESHOLD": hyperparameters.get("XCLIP_THRESHOLD"),
                "YOLO_MIN_THRESHOLD": hyperparameters.get("YOLO_MIN_THRESHOLD"),
                "MAX_NUMBER_OF_MODIFIED_PROMPTS": hyperparameters.get("MAX_NUMBER_OF_MODIFIED_PROMPTS"),
                "VIDEO_SAMPLING_RATE": hyperparameters.get("VIDEO_SAMPLING_RATE"),
                "hyperparameters": hyperparameters,
            }
        )
    return rows


def flatten_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        iou = row.get("iou", {})
        route_details = row.get("route_details", {})
        hyperparameters = row.get("hyperparameters", {})
        flattened.append(
            {
                "evaluated_at_utc": row.get("evaluated_at_utc"),
                "video": row.get("video"),
                "prompt_id": row.get("prompt_id"),
                "path_taken": row.get("path_taken"),
                "prompt": row.get("prompt"),
                "target_timestamps": row.get("target_timestamps", []),
                "predicted_timestamps": row.get("predicted_timestamps", []),
                "best_iou": iou.get("best_iou", 0.0),
                "mean_target_best_iou": iou.get("mean_target_best_iou", 0.0),
                "mean_predicted_best_iou": iou.get("mean_predicted_best_iou", 0.0),
                "recall": iou.get("recall", 0.0),
                "overlap_anywhere_recall": iou.get("overlap_anywhere_recall", 0.0),
                "temporal_set_iou": iou.get("temporal_set_iou", 0.0),
                "overlap_over_max": iou.get("overlap_over_max", 0.0),
                "duration_precision": iou.get("duration_precision", 0.0),
                "duration_recall": iou.get("duration_recall", 0.0),
                "overlap_duration_seconds": iou.get("overlap_duration_seconds", 0.0),
                "predicted_total_duration_seconds": iou.get("predicted_total_duration_seconds", 0.0),
                "target_total_duration_seconds": iou.get("target_total_duration_seconds", 0.0),
                "processing_seconds": row.get("processing_seconds", 0.0),
                "matched_frames_count": row.get("matched_frames_count", 0),
                "target_count": len(row.get("target_timestamps", [])),
                "predicted_count": len(row.get("predicted_timestamps", [])),
                "modified_prompt_count": len(route_details.get("modified_prompts", [])),
                "window_count": route_details.get("score_stats", {}).get("window_count"),
                "matched_window_count": route_details.get("score_stats", {}).get("matched_window_count"),
                "used_threshold": route_details.get("score_stats", {}).get("used_threshold"),
                "hyperparameter_combo_key": row.get("hyperparameter_combo_key"),
                "CLIP_THRESHOLD": hyperparameters.get("CLIP_THRESHOLD"),
                "XCLIP_THRESHOLD": hyperparameters.get("XCLIP_THRESHOLD"),
                "YOLO_MIN_THRESHOLD": hyperparameters.get("YOLO_MIN_THRESHOLD"),
            }
        )
    return flattened


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


st.set_page_config(page_title="Multimodal Retrieval Report", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(16,24,40,0.94), rgba(35,44,72,0.9));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.18);
    }
    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        color: #8ea6d8;
        margin-bottom: 0.25rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #9ca3af;
        margin-top: 0.15rem;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Multimodal Retrieval Report Explorer")
st.caption("Interactive view over evaluation reports, prompt-level outcomes, and hyperparameter comparisons.")

runs = load_all_runs()
cumulative_summary = load_json(CUMULATIVE_SUMMARY_PATH)
prompt_rows = load_jsonl(CUMULATIVE_RESULTS_PATH)
flattened_rows = flatten_prompt_rows(prompt_rows)

with st.sidebar:
    st.header("Report source")
    run_options = ["Cumulative report"] + [f"{run['run_id']} ({run['status']})" for run in runs]
    selected_run_label = st.selectbox("Dataset", run_options, index=0)
    st.divider()
    st.caption("Filters")
    available_paths = sorted({row.get("path_taken") for row in flattened_rows if row.get("path_taken")})
    selected_paths = st.multiselect("Path taken", options=available_paths, default=available_paths)
    min_iou, max_iou = st.slider("Best IoU range", 0.0, 1.0, (0.0, 1.0), 0.05)
    show_failed_only = st.checkbox("Show only recall failures", value=False)

if selected_run_label == "Cumulative report":
    active_summary = cumulative_summary.get("cumulative", {})
    path_rows = flatten_path_summary(cumulative_summary)
    combo_rows = flatten_combo_summary(cumulative_summary)
    run_meta = {
        "run_id": "cumulative",
        "status": "aggregated",
        "started_utc": cumulative_summary.get("last_updated_utc"),
        "finished_utc": cumulative_summary.get("last_updated_utc"),
    }
else:
    selected_run_id = selected_run_label.split(" ", 1)[0]
    selected_run = next((run for run in runs if run["run_id"] == selected_run_id), None)
    summary = (selected_run or {}).get("summary", {})
    active_summary = summary.get("summary", summary.get("cumulative", summary))
    path_rows = flatten_path_summary(summary)
    combo_rows = flatten_combo_summary(summary)
    run_meta = selected_run or {}

metrics = [
    ("Prompt count", active_summary.get("prompt_count", 0)),
    ("Avg best IoU", round(active_summary.get("avg_best_iou", 0.0), 4)),
    ("Avg recall", round(active_summary.get("avg_recall", 0.0), 4)),
    ("Avg overlap-anywhere recall", round(active_summary.get("avg_overlap_anywhere_recall", 0.0), 4)),
    ("Avg temporal set IoU", round(active_summary.get("avg_temporal_set_iou", 0.0), 4)),
    ("Avg overlap/max", round(active_summary.get("avg_overlap_over_max", 0.0), 4)),
    ("Total merged predicted s", round(active_summary.get("total_predicted_total_duration_seconds", 0.0), 2)),
    ("Total processing s", round(active_summary.get("total_processing_seconds", 0.0), 2)),
    ("Avg processing s", round(active_summary.get("avg_processing_seconds", 0.0), 2)),
    ("Avg duration precision", round(active_summary.get("avg_duration_precision", 0.0), 4)),
    ("Avg duration recall", round(active_summary.get("avg_duration_recall", 0.0), 4)),
]
metrics_per_row = 4
for start_index in range(0, len(metrics), metrics_per_row):
    chunk = metrics[start_index : start_index + metrics_per_row]
    cols = st.columns(len(chunk))
    for col, (label, value) in zip(cols, chunk):
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 1.5rem; font-weight: 700;'>{value}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.write("")
left_col, right_col = st.columns([1.15, 0.85])
with left_col:
    st.markdown("### Path performance")
    if path_rows:
        path_df = st.dataframe if False else None
        path_chart_df = sorted(path_rows, key=lambda row: row.get("prompt_count", 0), reverse=True)
        fig = px.bar(
            path_chart_df,
            x="path_taken",
            y=["avg_best_iou", "avg_recall"],
            barmode="group",
            color_discrete_sequence=[PATH_COLORS.get(path, "#4f83ff") for path in ["clip", "xclip", "yolo"]],
            title="Average IoU and recall by path",
        )
        fig.update_layout(yaxis_title="Score", xaxis_title="Path", legend_title_text="Metric")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No path summary data available in this report.")

with right_col:
    st.markdown("### Run status")
    start_dt = parse_timestamp(run_meta.get("started_utc"))
    finish_dt = parse_timestamp(run_meta.get("finished_utc"))
    duration_text = "n/a"
    if start_dt and finish_dt:
        duration_text = str(finish_dt - start_dt)
    status_box = {
        "Run ID": run_meta.get("run_id", "unknown"),
        "Status": run_meta.get("status", "unknown"),
        "Started": run_meta.get("started_utc", "n/a"),
        "Finished": run_meta.get("finished_utc", "n/a"),
        "Duration": duration_text,
    }
    st.json(status_box)
    if run_meta.get("last_error"):
        st.error("Last error captured in the run summary.")
        st.code(json.dumps(run_meta["last_error"], indent=2), language="json")

st.markdown("### Prompt-level results")
results_df = flattened_rows
if selected_paths:
    results_df = [row for row in results_df if row.get("path_taken") in selected_paths]
results_df = [row for row in results_df if min_iou <= float(row.get("best_iou", 0.0)) <= max_iou]
if show_failed_only:
    results_df = [row for row in results_df if float(row.get("recall", 0.0)) < 1.0]

if results_df:
    table_cols = [
        "evaluated_at_utc",
        "video",
        "prompt_id",
        "path_taken",
        "best_iou",
        "recall",
        "overlap_anywhere_recall",
        "temporal_set_iou",
        "overlap_over_max",
        "duration_precision",
        "duration_recall",
        "predicted_total_duration_seconds",
        "processing_seconds",
        "matched_frames_count",
        "modified_prompt_count",
        "hyperparameter_combo_key",
    ]
    st.dataframe(results_df, use_container_width=True, hide_index=True, column_order=table_cols)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        scatter = px.scatter(
            results_df,
            x="processing_seconds",
            y="best_iou",
            color="path_taken",
            hover_data=["video", "prompt_id", "hyperparameter_combo_key", "prompt"],
            color_discrete_map=PATH_COLORS,
            title="Processing time vs best IoU",
        )
        scatter.update_layout(xaxis_title="Processing seconds", yaxis_title="Best IoU")
        st.plotly_chart(scatter, use_container_width=True)
    with plot_col2:
        timeline_df = sorted(results_df, key=lambda row: row.get("evaluated_at_utc") or "")
        timeline = px.line(
            timeline_df,
            x="evaluated_at_utc",
            y="temporal_set_iou",
            color="path_taken",
            markers=True,
            color_discrete_map=PATH_COLORS,
            title="Temporal set IoU over evaluation order",
        )
        timeline.update_layout(xaxis_title="Evaluation timestamp", yaxis_title="Temporal set IoU")
        st.plotly_chart(timeline, use_container_width=True)
else:
    st.warning("No prompt rows matched the current filters.")

st.markdown("### Hyperparameter comparison")
if combo_rows:
    combo_df = combo_rows
    metric_choice = st.selectbox(
        "Sort combos by",
        ["avg_best_iou", "avg_recall", "avg_overlap_anywhere_recall", "avg_processing_seconds", "prompt_count"],
        index=0,
    )
    combo_df = sorted(combo_df, key=lambda row: float(row.get(metric_choice, 0.0)), reverse=(metric_choice != "avg_processing_seconds"))
    best_best_iou_combo = max(combo_df, key=lambda row: float(row.get("avg_best_iou", 0.0)))
    best_recall_combo = max(combo_df, key=lambda row: float(row.get("avg_recall", 0.0)))
    fastest_combo = min(combo_df, key=lambda row: float(row.get("avg_processing_seconds", 0.0)))

    top_cols = st.columns(3)
    with top_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Best avg IoU combo</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1.05rem; font-weight: 700;'>{best_best_iou_combo.get('hyperparameter_combo_key')}</div>", unsafe_allow_html=True)
        st.caption(f"avg_best_iou = {best_best_iou_combo.get('avg_best_iou'):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with top_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Best recall combo</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1.05rem; font-weight: 700;'>{best_recall_combo.get('hyperparameter_combo_key')}</div>", unsafe_allow_html=True)
        st.caption(f"avg_recall = {best_recall_combo.get('avg_recall'):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with top_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Fastest combo</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1.05rem; font-weight: 700;'>{fastest_combo.get('hyperparameter_combo_key')}</div>", unsafe_allow_html=True)
        st.caption(f"avg_processing_seconds = {fastest_combo.get('avg_processing_seconds'):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.dataframe(combo_df, use_container_width=True, hide_index=True)

    combo_options = [row.get("hyperparameter_combo_key") for row in combo_df if row.get("hyperparameter_combo_key")]
    selected_combo_key = st.selectbox("Inspect hyperparameter combo values", combo_options)
    selected_combo = next((row for row in combo_df if row.get("hyperparameter_combo_key") == selected_combo_key), None)
    if selected_combo:
        combo_left, combo_right = st.columns([0.45, 0.55])
        with combo_left:
            st.write("Selected combo summary")
            st.json(
                {
                    "hyperparameter_combo_key": selected_combo.get("hyperparameter_combo_key"),
                    "prompt_count": selected_combo.get("prompt_count"),
                    "avg_best_iou": selected_combo.get("avg_best_iou"),
                    "avg_recall": selected_combo.get("avg_recall"),
                    "avg_overlap_anywhere_recall": selected_combo.get("avg_overlap_anywhere_recall"),
                    "avg_processing_seconds": selected_combo.get("avg_processing_seconds"),
                    "total_predicted_total_duration_seconds": selected_combo.get("total_predicted_total_duration_seconds"),
                }
            )
        with combo_right:
            st.write("Full hyperparameter values")
            st.json(selected_combo.get("hyperparameters", {}))

    heatmap_rows: list[dict[str, Any]] = []
    for row in combo_df:
        heatmap_rows.append(
            {
                "combo": row.get("hyperparameter_combo_key"),
                "CLIP_THRESHOLD": row.get("CLIP_THRESHOLD"),
                "XCLIP_THRESHOLD": row.get("XCLIP_THRESHOLD"),
                "YOLO_MIN_THRESHOLD": row.get("YOLO_MIN_THRESHOLD"),
                "avg_best_iou": row.get("avg_best_iou"),
            }
        )
    if heatmap_rows:
        heatmap_fig = px.scatter(
            heatmap_rows,
            x="CLIP_THRESHOLD",
            y="XCLIP_THRESHOLD",
            size="avg_best_iou",
            color="avg_best_iou",
            hover_data=["combo", "YOLO_MIN_THRESHOLD"],
            title="Threshold exploration",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
else:
    st.info("No hyperparameter comparison data available in this report.")

st.markdown("### Per-video drilldown")
video_names = sorted({row.get("video") for row in results_df if row.get("video")})
selected_video = st.selectbox("Video", video_names if video_names else ["Jungle.mp4"])
video_runs = [row for row in results_df if row.get("video") == selected_video]
if video_runs:
    for row in video_runs:
        with st.expander(f"{row.get('prompt_id')} • {row.get('path_taken')} • IoU {row.get('best_iou'):.3f}", expanded=False):
            left, right = st.columns([1, 1])
            with left:
                st.write(row.get("prompt"))
                st.json(
                    {
                        "targets": row.get("target_timestamps"),
                        "predictions": row.get("predicted_timestamps"),
                        "processing_seconds": row.get("processing_seconds"),
                        "matched_frames_count": row.get("matched_frames_count"),
                        "modified_prompt_count": row.get("modified_prompt_count"),
                        "window_count": row.get("window_count"),
                        "overlap_duration_seconds": row.get("overlap_duration_seconds"),
                        "predicted_total_duration_seconds": row.get("predicted_total_duration_seconds"),
                        "target_total_duration_seconds": row.get("target_total_duration_seconds"),
                    }
                )
            with right:
                st.write("Route and IoU")
                st.json(
                    {
                        "best_iou": row.get("best_iou"),
                        "mean_target_best_iou": row.get("mean_target_best_iou"),
                        "mean_predicted_best_iou": row.get("mean_predicted_best_iou"),
                        "recall": row.get("recall"),
                        "overlap_anywhere_recall": row.get("overlap_anywhere_recall"),
                        "temporal_set_iou": row.get("temporal_set_iou"),
                        "overlap_over_max": row.get("overlap_over_max"),
                        "duration_precision": row.get("duration_precision"),
                        "duration_recall": row.get("duration_recall"),
                        "path_taken": row.get("path_taken"),
                        "combo_key": row.get("hyperparameter_combo_key"),
                    }
                )
else:
    st.info("No rows available for the selected video.")

st.markdown("### Raw files")
raw_path = st.selectbox(
    "Inspect file",
    [str(CUMULATIVE_SUMMARY_PATH)] + [str(CUMULATIVE_RESULTS_PATH)] + [str(run["run_dir"] / "run_summary.json") for run in runs],
)
if raw_path:
    raw_file = Path(raw_path)
    if raw_file.exists():
        st.code(raw_file.read_text(encoding="utf-8")[:20000], language="json")
