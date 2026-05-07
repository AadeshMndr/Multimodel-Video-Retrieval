"""
Charades-STA evaluation for the XCLIP sliding-window moment retrieval system.

Dataset : lmms-lab/charades_sta  (HuggingFace, test split)
Metric  : R@1 IoU ≥ {0.3, 0.5, 0.7}

Usage:
    python testing.py                          # full test split
    python testing.py --limit 50               # first 50 queries
    python testing.py --csv results.csv        # write per-query CSV
"""

import argparse
import csv
import logging
import os
import sys
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

VIDEO_BASE_PATH = "/home/aman/datasets/Charades_v1" #must have it locally 13GB, 15GB
IOU_THRESHOLDS = [0.3, 0.5, 0.7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def best_window_to_seconds(
    frame_ranges: list[tuple[int, int]],
    scores: list[float],
    fps: float,
) -> tuple[float, float]:
    """Return (start_sec, end_sec) for the highest-scoring window."""
    if len(scores) == 0:
        return 0.0, 0.0
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    start_frame, end_frame = frame_ranges[best_idx]
    return start_frame / fps, end_frame / fps


def build_frame_factory(video_processor: Video_Processor):
    """Replicate the frame-generator factory logic from the main pipeline."""
    sampled_factory = lambda: video_processor.sample_frames()
    if settings.XCLIP_USE_REDUCED_FRAMES:
        return lambda: video_processor.remove_similar_frames(sampled_factory())
    return sampled_factory


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(limit: int | None = None, csv_path: str | None = None):
    logging.info("Loading Charades-STA dataset from HuggingFace …")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    queries: list[dict] = []
    for row in ds:
        video_file = row["video"]
        video_path = os.path.join(VIDEO_BASE_PATH, video_file)
        gt_start, gt_end = row["timestamp"]
        queries.append({
            "video": video_file,
            "video_path": video_path,
            "caption": row["caption"],
            "gt_start": float(gt_start),
            "gt_end": float(gt_end),
        })

    if limit is not None:
        queries = queries[:limit]

    logging.info("Total queries to evaluate: %d", len(queries))

    queries_by_video: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        queries_by_video[q["video"]].append(q)

    logging.info("Unique videos: %d", len(queries_by_video))

    logging.info("Loading XCLIP model …")
    xclip_processor = XCLIP_Processor(
        model_name=settings.XCLIP_MODEL_NAME,
        embedding_store=None,
        device=settings.DEVICE,
    )

    hits: dict[float, int] = {t: 0 for t in IOU_THRESHOLDS}
    total = 0
    skipped = 0
    results: list[dict] = []

    video_list = sorted(queries_by_video.keys())
    wall_start = time.time()

    for vid_idx, video_file in enumerate(video_list, 1):
        video_queries = queries_by_video[video_file]
        video_path = video_queries[0]["video_path"]

        if not os.path.isfile(video_path):
            logging.warning("Video not found, skipping %d queries: %s", len(video_queries), video_path)
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

        frame_factory = build_frame_factory(vp)

        for q in video_queries:
            caption = q["caption"]
            gt_start, gt_end = q["gt_start"], q["gt_end"]

            try:
                frame_ranges, all_scores, _ = xclip_processor.compute_window_scores(
                    sampled_frames_factory=frame_factory,
                    texts=[caption],
                    fps=fps,
                )
            except Exception as exc:
                logging.warning("Score computation failed for %s / '%s': %s", video_file, caption, exc)
                skipped += 1
                continue

            pred_start, pred_end = best_window_to_seconds(frame_ranges, all_scores, fps)
            iou = compute_iou(pred_start, pred_end, gt_start, gt_end)

            for t in IOU_THRESHOLDS:
                if iou >= t:
                    hits[t] += 1

            total += 1

            results.append({
                "video": video_file,
                "caption": caption,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "pred_start": round(pred_start, 3),
                "pred_end": round(pred_end, 3),
                "iou": round(iou, 4),
                "hit@0.5": int(iou >= 0.5),
            })

        elapsed = time.time() - wall_start
        logging.info(
            "[%d/%d videos] %s  queries_so_far=%d  R@1_0.5=%.2f%%  elapsed=%.1fs",
            vid_idx, len(video_list), video_file,
            total,
            (hits[0.5] / total * 100) if total else 0,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    wall_elapsed = time.time() - wall_start
    print("\n" + "=" * 60)
    print("Charades-STA Evaluation Results")
    print("=" * 60)
    print(f"  Model            : {settings.XCLIP_MODEL_NAME}")
    print(f"  Window           : {settings.XCLIP_WINDOW_SECONDS}s  step {settings.XCLIP_STEP_SECONDS}s")
    print(f"  Frames/clip      : {settings.XCLIP_FRAMES_PER_CLIP}")
    print(f"  Sampling rate    : {settings.VIDEO_SAMPLING_RATE}")
    print(f"  Use reduced      : {settings.XCLIP_USE_REDUCED_FRAMES}")
    print(f"  Total queries    : {total}  (skipped {skipped})")
    print(f"  Unique videos    : {len(queries_by_video)}")
    for t in IOU_THRESHOLDS:
        pct = (hits[t] / total * 100) if total else 0
        print(f"  R@1 IoU≥{t:.1f}     : {hits[t]}/{total}  =  {pct:.2f}%")
    print(f"  Wall time        : {wall_elapsed:.1f}s")
    print("=" * 60)

    if csv_path:
        fieldnames = ["video", "caption", "gt_start", "gt_end",
                       "pred_start", "pred_end", "iou", "hit@0.5"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Per-query results written to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Charades-STA benchmark for XCLIP moment retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries")
    parser.add_argument("--csv", type=str, default=None, help="Path to write per-query CSV results")
    parser.add_argument("--video-dir", type=str, default=None,
                        help=f"Override video directory (default: {VIDEO_BASE_PATH})")
    args = parser.parse_args()

    global VIDEO_BASE_PATH
    if args.video_dir:
        VIDEO_BASE_PATH = args.video_dir #can point to a different directory if needed

    evaluate(limit=args.limit, csv_path=args.csv)


if __name__ == "__main__":
    main()


#python testing.py --csv charades_sta_results.csv 
