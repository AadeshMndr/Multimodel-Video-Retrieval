# Batch Evaluation Runner (Timestamp IoU + Recall)

Use `scripts/evaluate_retrieval.py` to batch-evaluate prompt-to-timestamp retrieval over a video folder + labels folder.

## What it does

- Loads videos from a folder.
- Loads prompts and manually labeled target timestamps from JSON files.
- Runs the existing retrieval workflow per prompt.
- Compares predicted timestamps vs target timestamps using interval IoU.
- Computes recall per prompt as `recall@IoU>=0.5` (fraction of target intervals matched by at least one prediction).
- Also computes `overlap_anywhere_recall`, a strict `IoU > 0.0` overlap-anywhere test.
- Captures video duration and per-prompt processing time.
- Writes:
  - **Per-run detailed reports** (including path chosen and route scores)
  - **Per-video detailed reports**
  - **Cumulative append-only prompt results**
  - **Cumulative summary**
- Avoids duplicate appends for prompts already evaluated earlier.

## Command

```bash
python scripts/evaluate_retrieval.py \
  --videos-dir evaluation/test_videos \
  --labels-dir evaluation/test_labels \
  --reports-dir evaluation/reports
```

All three directories default to the above values if omitted.

## Randomized Hyperparameter Search (Seeded)

To run seeded randomized search across `evaluation/hyperparameter_search_space.json`:

```bash
python scripts/evaluate_retrieval.py \
  --videos-dir evaluation/test_videos \
  --labels-dir evaluation/test_labels \
  --reports-dir evaluation/reports \
  --enable-random-search-cv \
  --max-random-combinations 20 \
  --random-seed 42
```

Notes:

- Default sampled combinations is `20` when random search is enabled.
- Using the same `--random-seed` and same search-space file yields the same sampled combinations.
- Results accumulate in `evaluation/reports/cumulative_prompt_results.jsonl` across days/runs.
- Dedup now includes both the prompt sample and the hyperparameter combination key, so the same prompt can be stored once per combination.
- Embedding stores are cleared when the runner switches to a new hyperparameter combination (default behavior).
- If you explicitly want to reuse embeddings between combinations, pass `--keep-embeddings-between-combinations`.

Additional run artifacts for random search:

- `runs/<run_id>/selected_hyperparameter_combinations.json`: sampled combinations for the run.
- `runs/<run_id>/embedding_clears.json`: which embedding files were removed before each combination.
- `runs/<run_id>/run_state.json`: live checkpoint state (completed count, last completed key, status, last error if any).

## Heat-Managed Cycle Runner (Run/Pause Loop)

Use `scripts/run_eval_in_cycles.sh` to run evaluation in cycles:

- run for N minutes,
- pause for M minutes,
- repeat until `evaluate_retrieval.py` exits naturally.

Default command executed each cycle:

```bash
python scripts/evaluate_retrieval.py --enable-random-search-cv --random-seed 42
```

Default timing:

- `RUN_MINUTES=10`
- `WAIT_MINUTES=10`

### Basic usage

```bash
scripts/run_eval_in_cycles.sh
```

### Configure run/pause in minutes

```bash
RUN_MINUTES=15 WAIT_MINUTES=5 scripts/run_eval_in_cycles.sh
```

### Override evaluator command

```bash
EVAL_CMD="python scripts/evaluate_retrieval.py --enable-random-search-cv --random-seed 42 --max-random-combinations 20" \
RUN_MINUTES=20 WAIT_MINUTES=10 \
scripts/run_eval_in_cycles.sh
```

### Optional advanced env vars

- `RUN_SECONDS`: second-based override for run window (takes precedence over `RUN_MINUTES`).
- `WAIT_SECONDS`: second-based override for pause window (takes precedence over `WAIT_MINUTES`).
- `MAX_GRACEFUL_STOP_SECONDS` (default `30`): how long to wait after SIGINT before SIGTERM fallback.

### Stop/resume behavior

- At the end of a run window, the cycle runner sends SIGINT to the evaluator (same signal as Ctrl+C) so checkpoint/resume logic is used.
- If evaluator does not stop within `MAX_GRACEFUL_STOP_SECONDS`, the runner sends SIGTERM as fallback.
- If evaluator exits with `0`, the cycle runner stops automatically (work completed).
- If evaluator exits non-zero, the cycle runner waits and retries next cycle.
- Pressing Ctrl+C on `run_eval_in_cycles.sh` stops the active evaluator and exits the loop cleanly.

## Resume After Interrupt or Error

The runner now persists each completed prompt result immediately to `cumulative_prompt_results.jsonl`.

This means:

- If you stop with Ctrl+C, completed prompts are not lost.
- If any runtime/API error occurs mid-run, completed prompts are not lost.
- Re-running the same command resumes from remaining prompts via dedup keys.

Recommended for day-by-day evaluation:

- Keep the same `--random-seed` and search-space file when random search is enabled.
- Do not use `--force-recompute` when you want resume behavior.

By default, this **does not create output clip videos**.

By default, recall is computed as `recall@IoU>=0.5`.

The reports also include `overlap_anywhere_recall`, which counts a target as found only when at least one predicted interval overlaps it with IoU `> 0.0`.

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

Added summary fields:

- `video_duration_seconds` per prompt record and per-video report.
- `processing_seconds` per prompt record.
- `total_processing_seconds` and `avg_processing_seconds` in summary sections.
- `overlap_anywhere_recall` per prompt record and `avg_overlap_anywhere_recall` in summary sections.
- `total_predicted_total_duration_seconds` in summary sections, representing the sum of merged predicted interval lengths across the summary scope.
- `summary_by_path` in run summary, per-video reports, and cumulative summary.

## Dedup behavior across runs

Each prompt sample is assigned a stable key from:

- video path,
- prompt text,
- target timestamp labels.
- hyperparameter combination key (or `default` when random search is disabled).

If that key already exists in `cumulative_prompt_results.jsonl`, it is skipped and not appended again.

Use `--force-recompute` to run/append regardless of existing keys.
