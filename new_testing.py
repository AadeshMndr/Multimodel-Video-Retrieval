"""
Charades-STA evaluation - extended metrics version (per-video records).

This is a SEPARATE pipeline from `testing.py`. The original R@1 IoU (per-query
hit-rate at thresholds 0.3 / 0.5 / 0.7) is preserved exactly as the global
summary. In addition, the following per-record metrics are computed and
averaged across records (record = one video):

    - avg_recall                     (greedy match @ IoU >= 0.3 / 0.5 / 0.7)
    - avg_best_iou
    - avg_mean_target_best_iou
    - avg_mean_predicted_best_iou
    - avg_overlap_anywhere_recall    (greedy match @ IoU > 0)
    - avg_temporal_set_iou           (temporal Jaccard of merged sets)
    - avg_overlap_over_max
    - avg_duration_precision
    - avg_duration_recall
    - total_predicted_total_duration_seconds  (SUM, not average)

In addition, the script now computes query-level AP from the ranked sliding
windows and reports mean AP (mAP) across IoU thresholds 0.3 / 0.5 / 0.7.

Definitions
-----------
- Record               : one video (queries grouped by video).
- Targets per record   : list of (gt_start, gt_end) intervals across all
                         queries on that video.
- Predictions per rec. : ALL sliding-window intervals returned by
                         `XCLIP_Processor.compute_window_scores` for the
                         video (caption-agnostic, hence shared across the
                         video's queries).
- Matching             : greedy by descending IoU. Each target and each
                         prediction may be used at most once.
- Division by zero     : 0 / 0 := 0.0 for all ratio metrics.
- Empty record         : a video with zero targets or zero predictions
                         contributes 0.0 to all ratio metrics (still
                         counted in the average denominator), and its
                         predicted_total_duration_seconds still feeds the
                         total sum.

Usage:
    python new_testing.py
    python new_testing.py --limit 50
    python new_testing.py --csv charades_sta_extended.csv
"""

import argparse
import csv
import logging
import os
import time
from collections import defaultdict

from datasets import load_dataset

from config import settings
from infrastructure.video_processor import Video_Processor
from infrastructure.xclip_processor import XCLIP_Processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

USE_personal_DATASET = False
if USE_personal_DATASET:
    DATASET_NAME = "my_dataset/myDataset"
    DATASET_SPLIT = "test"
else:
    DATASET_NAME = "lmms-lab/charades_sta"
    DATASET_SPLIT = "test"

VIDEO_BASE_PATH = "/Users/aadeshmanandhar/Downloads/Charades_v1_480"
IOU_THRESHOLDS = [0.3, 0.5, 0.7]
EPS = 1e-9


# ---------------------------------------------------------------------------
# IoU and interval helpers
# ---------------------------------------------------------------------------

def compute_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def safe_div(numerator: float, denominator: float) -> float:
    """0 / 0 := 0.0 by convention for all ratio metrics."""
    if denominator <= EPS:
        return 0.0
    return numerator / denominator


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping / touching intervals; drop zero-length ones."""
    cleaned = [(s, e) for s, e in intervals if e > s]
    if not cleaned:
        return []
    cleaned.sort()
    merged: list[list[float]] = [list(cleaned[0])]
    for s, e in cleaned[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def total_duration(intervals: list[tuple[float, float]]) -> float:
    """Total seconds covered by the union of intervals (merged)."""
    return sum(e - s for s, e in merge_intervals(intervals))


def intersection_duration(
    a_intervals: list[tuple[float, float]],
    b_intervals: list[tuple[float, float]],
) -> float:
    """Total seconds covered by both A and B as merged sets."""
    a_merged = merge_intervals(a_intervals)
    b_merged = merge_intervals(b_intervals)
    if not a_merged or not b_merged:
        return 0.0
    total = 0.0
    j_start = 0
    for a_s, a_e in a_merged:
        for j in range(j_start, len(b_merged)):
            b_s, b_e = b_merged[j]
            if b_e <= a_s:
                j_start = j + 1
                continue
            if b_s >= a_e:
                break
            total += min(a_e, b_e) - max(a_s, b_s)
    return total


def best_window_to_seconds(
    frame_ranges: list[tuple[int, int]],
    scores: list[float],
    fps: float,
) -> tuple[float, float]:
    if len(scores) == 0:
        return 0.0, 0.0
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    sf, ef = frame_ranges[best_idx]
    return sf / fps, ef / fps


def average_precision_at_iou(
    frame_ranges: list[tuple[int, int]],
    scores: list[float],
    gt_start: float,
    gt_end: float,
    fps: float,
    threshold: float,
) -> float:
    """Average precision for a single query at one IoU threshold.

    Each sliding window is treated as a ranked prediction. A window is relevant
    if its IoU with the ground-truth interval is at least `threshold`.
    """
    if not frame_ranges or not scores:
        return 0.0

    if fps <= 0:
        return 0.0

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    relevant_total = 0
    precision_sum = 0.0

    for rank, idx in enumerate(ranked_indices, start=1):
        window_start, window_end = frame_ranges[idx]
        iou = compute_iou(window_start / fps, window_end / fps, gt_start, gt_end)
        if iou >= threshold:
            relevant_total += 1
            precision_sum += relevant_total / rank

    if relevant_total == 0:
        return 0.0
    return precision_sum / relevant_total


def build_frame_factory(video_processor: Video_Processor):
    sampled_factory = lambda: video_processor.sample_frames()
    if settings.XCLIP_USE_REDUCED_FRAMES:
        return lambda: video_processor.remove_similar_frames(sampled_factory())
    return sampled_factory


# ---------------------------------------------------------------------------
# Per-record metric computation
# ---------------------------------------------------------------------------

def compute_pairwise_iou(
    targets: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
) -> list[list[float]]:
    """iou[i][j] = IoU(targets[i], predictions[j])."""
    return [
        [compute_iou(ts, te, ps, pe) for ps, pe in predictions]
        for ts, te in targets
    ]


def greedy_match(iou_matrix: list[list[float]], threshold: float) -> set[int]:
    """
    Greedy bipartite matching by descending IoU.
    Returns the set of matched target indices.
    Each target and each prediction is used at most once.
    """
    if not iou_matrix or not iou_matrix[0]:
        return set()
    pairs = []
    for i, row in enumerate(iou_matrix):
        for j, v in enumerate(row):
            if v >= threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_t: set[int] = set()
    used_p: set[int] = set()
    matched: set[int] = set()
    for _, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matched.add(i)
    return matched


def record_metrics(
    targets: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
    recall_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict:
    """All per-record metric values for a single (record, predictions) pair."""
    n_t = len(targets)
    n_p = len(predictions)

    out = {
        "n_targets": n_t,
        "n_predictions": n_p,
        "best_iou": 0.0,
        "mean_target_best_iou": 0.0,
        "mean_predicted_best_iou": 0.0,
        "overlap_anywhere_recall": 0.0,
        "temporal_set_iou": 0.0,
        "overlap_over_max": 0.0,
        "duration_precision": 0.0,
        "duration_recall": 0.0,
        "predicted_total_duration_seconds": total_duration(predictions),
    }
    for t in recall_thresholds:
        out[f"recall_{t}"] = 0.0

    if n_t == 0 or n_p == 0:
        return out

    iou_mat = compute_pairwise_iou(targets, predictions)

    out["best_iou"] = max(max(row) for row in iou_mat)
    out["mean_target_best_iou"] = sum(max(row) for row in iou_mat) / n_t
    out["mean_predicted_best_iou"] = sum(
        max(iou_mat[i][j] for i in range(n_t)) for j in range(n_p)
    ) / n_p

    out["overlap_anywhere_recall"] = len(greedy_match(iou_mat, EPS)) / n_t
    for t in recall_thresholds:
        out[f"recall_{t}"] = len(greedy_match(iou_mat, t)) / n_t

    target_dur = total_duration(targets)
    pred_dur = out["predicted_total_duration_seconds"]
    overlap_dur = intersection_duration(targets, predictions)

    out["temporal_set_iou"] = safe_div(
        overlap_dur, target_dur + pred_dur - overlap_dur
    )
    out["overlap_over_max"] = safe_div(overlap_dur, max(target_dur, pred_dur))
    out["duration_precision"] = safe_div(overlap_dur, pred_dur)
    out["duration_recall"] = safe_div(overlap_dur, target_dur)

    return out


# ---------------------------------------------------------------------------
# Main evaluation (record = video)
# ---------------------------------------------------------------------------

def evaluate(limit: int | None = None, csv_path: str | None = None, offset: int = 0, processing_csv: str | None = None):
    logging.info("Loading Charades-STA dataset from HuggingFace …")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    queries: list[dict] = []
    for row in ds:
        video_file = row["video"]
        gt_start, gt_end = row["timestamp"]
        queries.append({
            "video": video_file,
            "video_path": os.path.join(VIDEO_BASE_PATH, video_file),
            "caption": row["caption"],
            "gt_start": float(gt_start),
            "gt_end": float(gt_end),
        })

    total_in_dataset = len(queries)
    offset = max(0, offset)
    end = offset + limit if limit is not None else None
    queries = queries[offset:end]
    # assign a global prompt id (useful for tracking processing time rows)
    for idx, q in enumerate(queries, start=offset):
        q["prompt_id"] = idx
    logging.info(
        "Slice: offset=%d  limit=%s  -> evaluating %d / %d queries",
        offset, str(limit), len(queries), total_in_dataset,
    )

    queries_by_video: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        queries_by_video[q["video"]].append(q)

    logging.info("Unique videos (records): %d", len(queries_by_video))

    logging.info("Loading XCLIP model …")
    xclip_processor = XCLIP_Processor(
        model_name=settings.XCLIP_MODEL_NAME,
        embedding_store=None,
        device=settings.DEVICE,
    )

    # Prepare processing-time CSV (one per run) if not supplied
    if processing_csv is None:
        ts = int(time.time())
        os.makedirs("Processing_time_results", exist_ok=True)
        processing_csv = os.path.join("Processing_time_results", f"processing_time_{ts}.csv")
    else:
        # If the user supplied a path that already exists, create a new unique file
        # by appending timestamp and pid to avoid accidental appends.
        if os.path.exists(processing_csv):
            base, ext = os.path.splitext(processing_csv)
            ts = int(time.time())
            processing_csv = f"{base}_{ts}_{os.getpid()}{ext}"
        parent = os.path.dirname(processing_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)

    proc_fieldnames = [
        "video",
        "prompt_id",
        "prompt_in_video_id",
        "original_video_duration_seconds",
        "processing_time_seconds",
        "gt_start",
        "gt_end",
    ]
    proc_file = open(processing_csv, "w", newline="", encoding="utf-8")
    proc_writer = csv.DictWriter(proc_file, fieldnames=proc_fieldnames)
    proc_writer.writeheader()

    # R@1 globals (existing metric, kept exactly as in testing.py)
    hits: dict[float, int] = {t: 0 for t in IOU_THRESHOLDS}
    ap_sums: dict[float, float] = {t: 0.0 for t in IOU_THRESHOLDS}
    total_queries = 0
    skipped = 0

    record_rows: list[dict] = []
    total_predicted_total_duration_seconds = 0.0

    video_list = sorted(queries_by_video.keys())
    wall_start = time.time()

    for vid_idx, video_file in enumerate(video_list, 1):
        video_queries = queries_by_video[video_file]
        video_path = video_queries[0]["video_path"]

        if not os.path.isfile(video_path):
            logging.warning("Video not found, skipping %d queries: %s",
                            len(video_queries), video_path)
            skipped += len(video_queries)
            continue

        try:
            vp = Video_Processor(video_path)
        except Exception as exc:
            logging.warning("Cannot open video %s: %s", video_path, exc)
            skipped += len(video_queries)
            continue

        fps = vp.fps
        if fps <= 0:
            logging.warning("Invalid fps for %s, skipping", video_path)
            skipped += len(video_queries)
            continue

        original_video_duration_seconds = vp.frame_count / fps

        frame_factory = build_frame_factory(vp)

        targets: list[tuple[float, float]] = []
        predictions: list[tuple[float, float]] | None = None
        video_ap_sums: dict[float, float] = {t: 0.0 for t in IOU_THRESHOLDS}
        video_query_count = 0

        for q_idx, q in enumerate(video_queries):
            caption = q["caption"]
            gt_start, gt_end = q["gt_start"], q["gt_end"]

            start_pt = time.perf_counter()
            try:
                frame_ranges, all_scores, _ = xclip_processor.compute_window_scores(
                    sampled_frames_factory=frame_factory,
                    texts=[caption],
                    fps=fps,
                )
                processing_time = time.perf_counter() - start_pt
            except Exception as exc:
                logging.warning("Score computation failed for %s / '%s': %s",
                                video_file, caption, exc)
                skipped += 1
                processing_time = ""

            # R@1 (per-query, identical to testing.py)
            pred_start, pred_end = best_window_to_seconds(frame_ranges, all_scores, fps)
            iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
            for t in IOU_THRESHOLDS:
                if iou >= t:
                    hits[t] += 1
                ap_value = average_precision_at_iou(
                    frame_ranges,
                    all_scores,
                    gt_start,
                    gt_end,
                    fps,
                    t,
                )
                ap_sums[t] += ap_value
                video_ap_sums[t] += ap_value
            total_queries += 1
            video_query_count += 1

            targets.append((gt_start, gt_end))

            # frame_ranges are caption-agnostic; capture once per video.
            if predictions is None and frame_ranges:
                predictions = [(sf / fps, ef / fps) for sf, ef in frame_ranges]

            # write processing-time row for this prompt
            proc_writer.writerow({
                "video": video_file,
                "prompt_id": q.get("prompt_id"),
                "prompt_in_video_id": q_idx,
                "original_video_duration_seconds": original_video_duration_seconds,
                "processing_time_seconds": processing_time,
                "gt_start": gt_start,
                "gt_end": gt_end,
            })

        if not targets:
            continue

        rec = record_metrics(targets, predictions or [])
        rec["video"] = video_file
        rec["original_video_duration_seconds"] = original_video_duration_seconds
        for t in IOU_THRESHOLDS:
            rec[f"ap_{t}"] = safe_div(video_ap_sums[t], video_query_count)
        rec["mAP"] = sum(rec[f"ap_{t}"] for t in IOU_THRESHOLDS) / len(IOU_THRESHOLDS)
        record_rows.append(rec)
        total_predicted_total_duration_seconds += rec["predicted_total_duration_seconds"]

        elapsed = time.time() - wall_start
        logging.info(
            "[%d/%d] %s  T=%d P=%d  best_iou=%.3f  set_iou=%.3f  "
            "recall@0.5=%.3f  R@1_0.5=%.2f%%  elapsed=%.1fs",
            vid_idx, len(video_list), video_file,
            rec["n_targets"], rec["n_predictions"],
            rec["best_iou"], rec["temporal_set_iou"], rec["recall_0.5"],
            (hits[0.5] / total_queries * 100) if total_queries else 0,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------
    n = len(record_rows)

    def avg(field: str) -> float:
        if n == 0:
            return 0.0
        return sum(r[field] for r in record_rows) / n

    summary = {
        "n_records": n,
        "total_queries": total_queries,
        "skipped": skipped,
        "avg_best_iou": avg("best_iou"),
        "avg_mean_target_best_iou": avg("mean_target_best_iou"),
        "avg_mean_predicted_best_iou": avg("mean_predicted_best_iou"),
        "avg_overlap_anywhere_recall": avg("overlap_anywhere_recall"),
        "avg_recall_0.3": avg("recall_0.3"),
        "avg_recall_0.5": avg("recall_0.5"),
        "avg_recall_0.7": avg("recall_0.7"),
        "avg_temporal_set_iou": avg("temporal_set_iou"),
        "avg_overlap_over_max": avg("overlap_over_max"),
        "avg_duration_precision": avg("duration_precision"),
        "avg_duration_recall": avg("duration_recall"),
        "total_predicted_total_duration_seconds": total_predicted_total_duration_seconds,
    }

    for t in IOU_THRESHOLDS:
        summary[f"avg_ap_{t}"] = safe_div(ap_sums[t], total_queries)
    summary["mAP"] = sum(summary[f"avg_ap_{t}"] for t in IOU_THRESHOLDS) / len(IOU_THRESHOLDS)

    wall_elapsed = time.time() - wall_start
    print("\n" + "=" * 64)
    print("Charades-STA Extended Evaluation (per-video records)")
    print("=" * 64)
    print(f"  Model            : {settings.XCLIP_MODEL_NAME}")
    print(f"  Window           : {settings.XCLIP_WINDOW_SECONDS}s  step {settings.XCLIP_STEP_SECONDS}s")
    print(f"  Frames/clip      : {settings.XCLIP_FRAMES_PER_CLIP}")
    print(f"  Sampling rate    : {settings.VIDEO_SAMPLING_RATE}")
    print(f"  Use reduced      : {settings.XCLIP_USE_REDUCED_FRAMES}")
    print(f"  Records (videos) : {n}")
    print(f"  Total queries    : {total_queries}  (skipped {skipped})")
    print("-" * 64)
    print("R@1 IoU (existing per-query metric, unchanged):")
    for t in IOU_THRESHOLDS:
        pct = (hits[t] / total_queries * 100) if total_queries else 0
        print(f"  R@1 IoU>={t:.1f}    : {hits[t]}/{total_queries}  =  {pct:.2f}%")
    print("-" * 64)
    print("mAP (query-level AP over ranked windows):")
    for t in IOU_THRESHOLDS:
        print(f"  AP@{t:.1f}                    : {summary[f'avg_ap_{t}']:.4f}")
    print(f"  mAP                        : {summary['mAP']:.4f}")
    print("-" * 64)
    print("Extended metrics (averaged across records):")
    print(f"  avg_recall@0.3                  : {summary['avg_recall_0.3']:.4f}")
    print(f"  avg_recall@0.5                  : {summary['avg_recall_0.5']:.4f}")
    print(f"  avg_recall@0.7                  : {summary['avg_recall_0.7']:.4f}")
    print(f"  avg_overlap_anywhere_recall     : {summary['avg_overlap_anywhere_recall']:.4f}")
    print(f"  avg_best_iou                    : {summary['avg_best_iou']:.4f}")
    print(f"  avg_mean_target_best_iou        : {summary['avg_mean_target_best_iou']:.4f}")
    print(f"  avg_mean_predicted_best_iou     : {summary['avg_mean_predicted_best_iou']:.4f}")
    print(f"  avg_temporal_set_iou            : {summary['avg_temporal_set_iou']:.4f}")
    print(f"  avg_overlap_over_max            : {summary['avg_overlap_over_max']:.4f}")
    print(f"  avg_duration_precision          : {summary['avg_duration_precision']:.4f}")
    print(f"  avg_duration_recall             : {summary['avg_duration_recall']:.4f}")
    print(f"  total_predicted_total_duration  : {summary['total_predicted_total_duration_seconds']:.2f}s")
    print(f"  Wall time                       : {wall_elapsed:.1f}s")
    print("=" * 64)

    # ------------------------------------------------------------------
    # CSV: per-record rows + a single trailing __SUMMARY__ row.
    # ------------------------------------------------------------------
    if csv_path:
        fieldnames = [
            "video", "original_video_duration_seconds", "n_targets", "n_predictions",
            "recall_0.3", "recall_0.5", "recall_0.7",
            "overlap_anywhere_recall",
            "best_iou", "mean_target_best_iou", "mean_predicted_best_iou",
            "temporal_set_iou", "overlap_over_max",
            "duration_precision", "duration_recall",
            "ap_0.3", "ap_0.5", "ap_0.7", "mAP",
            "predicted_total_duration_seconds",
        ]

        def fmt(row: dict) -> dict:
            return {
                k: (round(row[k], 6) if isinstance(row[k], float) else row[k])
                for k in fieldnames
            }

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in record_rows:
                writer.writerow(fmt(r))

            summary_row = {f: "" for f in fieldnames}
            summary_row["video"] = "__SUMMARY__"
            summary_row["original_video_duration_seconds"] = ""
            summary_row["recall_0.3"] = round(summary["avg_recall_0.3"], 6)
            summary_row["recall_0.5"] = round(summary["avg_recall_0.5"], 6)
            summary_row["recall_0.7"] = round(summary["avg_recall_0.7"], 6)
            summary_row["overlap_anywhere_recall"] = round(summary["avg_overlap_anywhere_recall"], 6)
            summary_row["best_iou"] = round(summary["avg_best_iou"], 6)
            summary_row["mean_target_best_iou"] = round(summary["avg_mean_target_best_iou"], 6)
            summary_row["mean_predicted_best_iou"] = round(summary["avg_mean_predicted_best_iou"], 6)
            summary_row["temporal_set_iou"] = round(summary["avg_temporal_set_iou"], 6)
            summary_row["overlap_over_max"] = round(summary["avg_overlap_over_max"], 6)
            summary_row["duration_precision"] = round(summary["avg_duration_precision"], 6)
            summary_row["duration_recall"] = round(summary["avg_duration_recall"], 6)
            summary_row["ap_0.3"] = round(summary["avg_ap_0.3"], 6)
            summary_row["ap_0.5"] = round(summary["avg_ap_0.5"], 6)
            summary_row["ap_0.7"] = round(summary["avg_ap_0.7"], 6)
            summary_row["mAP"] = round(summary["mAP"], 6)
            summary_row["predicted_total_duration_seconds"] = round(
                summary["total_predicted_total_duration_seconds"], 6
            )
            writer.writerow(summary_row)

        print(f"Per-record results written to {csv_path}")

    # Close processing-time CSV file (if opened)
    try:
        proc_file.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global VIDEO_BASE_PATH

    parser = argparse.ArgumentParser(
        description="Charades-STA extended-metrics benchmark (per-video records)"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only N queries (after offset)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N queries before applying --limit. "
                             "Use with a different --offset in another process to run "
                             "disjoint slices in parallel (e.g. offset=0 limit=500 vs offset=500 limit=500).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to write per-record CSV results")
    parser.add_argument("--video-dir", type=str, default=None,
                        help=f"Override video directory (default: {VIDEO_BASE_PATH})")
    parser.add_argument("--processing-csv", type=str, default=None,
                        help="Path to write per-prompt processing times CSV (default: Processing_time_results/processing_time_<ts>.csv)")
    args = parser.parse_args()

    if args.video_dir:
        VIDEO_BASE_PATH = args.video_dir

    evaluate(limit=args.limit, csv_path=args.csv, offset=args.offset, processing_csv=args.processing_csv)


if __name__ == "__main__":
    main()


# python new_testing.py --csv charades_sta_extended.csv
