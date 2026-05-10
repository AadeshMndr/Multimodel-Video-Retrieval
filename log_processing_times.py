#!/usr/bin/env python3
"""Measure per-prompt processing time and video duration.

Writes a CSV to `Processing_time_results/processing_time_<ts>.csv` with columns:
  video, prompt_id, prompt_in_video_id, original_video_duration_seconds,
  processing_time_seconds, gt_start, gt_end

Usage:
  python log_processing_times.py --limit 100 --offset 0
"""
import argparse
import csv
import os
import time
from collections import defaultdict

from datasets import load_dataset

from config import settings
from infrastructure.video_processor import Video_Processor
from infrastructure.xclip_processor import XCLIP_Processor


def parse_args():
    p = argparse.ArgumentParser(description="Log processing time per prompt")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--csv", type=str, default=None, help="Output CSV path")
    p.add_argument("--video-dir", type=str, default=None)
    return p.parse_args()


def ensure_out_dir(path: str) -> str:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return path


def main():
    args = parse_args()

    if args.video_dir:
        global VIDEO_BASE_PATH
        VIDEO_BASE_PATH = args.video_dir

    ds = load_dataset("lmms-lab/charades_sta", split="test")

    queries = []
    for row in ds:
        video_file = row["video"]
        gt_start, gt_end = row["timestamp"]
        queries.append({
            "video": video_file,
            "video_path": os.path.join(VIDEO_BASE_PATH, video_file),
            "gt_start": float(gt_start),
            "gt_end": float(gt_end),
        })

    total = len(queries)
    offset = max(0, args.offset)
    end = offset + args.limit if args.limit is not None else None
    queries = queries[offset:end]

    # group by video
    queries_by_video = defaultdict(list)
    for idx, q in enumerate(queries, start=offset):
        q["prompt_id"] = idx
        queries_by_video[q["video"]].append(q)

    xclip = XCLIP_Processor(model_name=settings.XCLIP_MODEL_NAME, embedding_store=None, device=settings.DEVICE)

    ts = int(time.time())
    out_dir = "Processing_time_results"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = args.csv or os.path.join(out_dir, f"processing_time_{ts}.csv")
    ensure_out_dir(out_csv)

    fieldnames = [
        "video",
        "prompt_id",
        "prompt_in_video_id",
        "original_video_duration_seconds",
        "processing_time_seconds",
        "gt_start",
        "gt_end",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for video_file, qlist in sorted(queries_by_video.items()):
            if not qlist:
                continue
            video_path = qlist[0]["video_path"]
            if not os.path.isfile(video_path):
                continue
            try:
                vp = Video_Processor(video_path)
            except Exception:
                continue

            fps = vp.fps
            if fps <= 0:
                continue

            original_duration = vp.frame_count / fps
            frame_factory = lambda: vp.sample_frames()

            for i, q in enumerate(qlist):
                caption = ""  # do not store caption text here
                gt_start = q["gt_start"]
                gt_end = q["gt_end"]
                prompt_id = q["prompt_id"]

                start_time = time.perf_counter()
                try:
                    # measure the core scoring call
                    _, _, _ = xclip.compute_window_scores(sampled_frames_factory=frame_factory, texts=[caption], fps=fps)
                except Exception:
                    processing_time = None
                else:
                    processing_time = time.perf_counter() - start_time

                row = {
                    "video": video_file,
                    "prompt_id": prompt_id,
                    "prompt_in_video_id": i,
                    "original_video_duration_seconds": original_duration,
                    "processing_time_seconds": processing_time if processing_time is not None else "",
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                }
                writer.writerow(row)

    print(f"Processing time CSV written to {out_csv}")


if __name__ == "__main__":
    # default VIDEO_BASE_PATH is set in other scripts; declare here for override
    VIDEO_BASE_PATH = "/Users/aadeshmanandhar/Downloads/Charades_v1_480"
    main()
