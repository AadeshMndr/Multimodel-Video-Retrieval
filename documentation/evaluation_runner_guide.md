# Batch Evaluation Runner (Timestamp IoU + Recall)

Use `scripts/evaluate_retrieval.py` to batch-evaluate prompt-to-timestamp retrieval over a video folder + labels folder.

## What it does

- Loads videos from a folder.
- Loads prompts and manually labeled target timestamps from JSON files.
- Runs the existing retrieval workflow per prompt.
- Compares predicted timestamps vs target timestamps using interval IoU.
- Computes recall per prompt as `recall@IoU>=0.5` (fraction of target intervals matched by at least one prediction).
- Writes:
  - **Per-run detailed reports** (including path chosen and route scores)
  - **Per-video detailed reports**
  - **Cumulative append-only prompt results**
  - **Cumulative summary**
- Avoids duplicate appends for prompts already evaluated earlier.

## Command

```bash
python scripts/evaluate_retrieval.py \
  --videos-dir video_storage \
  --labels-dir documentation/eval_labels \
  --reports-dir outputs/evaluation
```

By default, this **does not create output clip videos**.

By default, recall is computed as `recall@IoU>=0.5`.

To change the recall IoU threshold, add:

```bash
--recall-iou-threshold 0.3
```

If you want output videos as well, add:

```bash
--generate-video
```

## Label file format

Create one JSON file per video prompt-set. The file can optionally include `video`; otherwise, the script matches a video with the same stem as the label file.

```json
{
  "video": "my_video.mp4",
  "prompts": [
    {
      "id": "p1",
      "prompt": "person drinking water",
      "target_timestamp": [12, 19]
    },
    {
      "id": "p2",
      "prompt": "person opening door",
      "target_timestamps": [[31, 36], [40, 43]]
    }
  ]
}
```

`target_timestamp` and `target_timestamps` are both supported.

## Report outputs

Inside `--reports-dir`:

- `runs/<run_id>/run_summary.json`: summary of the current run.
- `runs/<run_id>/per_video/<video>.json`: detailed per-video report with per-prompt scores.
- `cumulative_prompt_results.jsonl`: append-only records for prompts across runs.
- `cumulative_summary.json`: combined metrics across all appended runs.

## Dedup behavior across runs

Each prompt sample is assigned a stable key from:

- video path,
- prompt text,
- target timestamp labels.

If that key already exists in `cumulative_prompt_results.jsonl`, it is skipped and not appended again.

Use `--force-recompute` to run/append regardless of existing keys.
