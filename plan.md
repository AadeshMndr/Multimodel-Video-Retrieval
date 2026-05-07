# XCLIP Moment Retrieval — Charades-STA Evaluation Plan

## Objective

Evaluate the XCLIP sliding-window moment retrieval system on the
**Charades-STA** benchmark (`lmms-lab/charades_sta` on HuggingFace, test split,
3 720 query–video pairs) using the standard **R@1 IoU ≥ 0.5** metric.

## Dataset

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| `video`     | Filename, e.g. `3MSZA.mp4`                      |
| `caption`   | Natural-language query                           |
| `timestamp` | Ground-truth moment as `[start_sec, end_sec]`    |

Videos live at `/home/aman/datasets/Charades_v1/`.

## Pipeline (per query)

1. **Load video metadata** — `Video_Processor(video_path)` to get `fps`,
   `frame_count`.
2. **Build frame generators** — `sample_frames()` (every `VIDEO_SAMPLING_RATE`
   frames) → optionally `remove_similar_frames()` (controlled by
   `XCLIP_USE_REDUCED_FRAMES`).
3. **Score windows** — `XCLIP_Processor.compute_window_scores()` returns
   `frame_ranges` (list of `(start_frame, end_frame)`) and `all_scores`.
4. **Predict moment** — take the window with the highest score. Convert its
   frame range to seconds: `start_sec = start_frame / fps`,
   `end_sec = end_frame / fps`.
5. **Compute IoU** — `IoU = intersection / union` between predicted and
   ground-truth intervals.
6. **Accumulate** — a query is a hit when IoU ≥ 0.5.

## Optimisations

- **Per-video caching**: group queries by `video` so the XCLIP model loads
  window embeddings only once per unique video, then scores multiple captions
  against the cached embeddings.
- **Embedding store disabled** for benchmark runs (no HDF5 side-effects).
- **No LLM prompt variation** — use the raw caption directly to measure the
  base XCLIP retrieval quality.

## Metric

```
R@1 IoU≥0.5 = (# queries with IoU ≥ 0.5) / (total queries) × 100 %
```

Additionally report R@1 IoU ≥ 0.3 and R@1 IoU ≥ 0.7 for a fuller picture.

## Output

- Per-query CSV log (`charades_sta_results.csv`): video, caption,
  gt_start, gt_end, pred_start, pred_end, iou, hit@0.5.
- Summary printed to stdout at the end.
