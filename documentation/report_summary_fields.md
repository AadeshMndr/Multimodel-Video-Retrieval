# Report Summary Fields

This document explains the main fields that appear in the evaluation report files under `evaluation/reports/`.

The same metric names can appear in both a single-run summary and the cumulative summary. The difference is scope:

- `run_summary.json` describes one evaluation run.
- `cumulative_summary.json` aggregates across all runs that have been collected so far.
- `cumulative_prompt_results.jsonl` stores one row per evaluated prompt/video/hyperparameter combination.
- `per_video/*.json` contains the same kind of summary, but grouped by video.

## High-level summary fields

### `prompt_count`
The number of evaluated prompt-task records included in the summary.

For a single run, this is how many prompt evaluations were completed in that run.
For the cumulative summary, this is the total number of prompt evaluations across all stored runs.

### `avg_best_iou`
The average of the best IoU value for each evaluated record.

For each prompt evaluation, the system may predict multiple timestamp intervals. The best IoU is the highest Intersection over Union between any predicted interval and the ground-truth interval(s) for that prompt.

This field answers: "On average, how well did the best predicted match align with the target?"

Averaging scope:

- It is averaged over records (rows), not over timestamps directly.
- A record means one evaluated prompt/video/hyperparameter result (one line in `cumulative_prompt_results.jsonl`).

Example:

- Record A has `best_iou = 0.40`
- Record B has `best_iou = 0.70`
- Record C has `best_iou = 0.10`

Then:

$$
avg\_best\_iou = \frac{0.40 + 0.70 + 0.10}{3} = 0.40
$$

### `avg_mean_target_best_iou`
For each target interval, the system finds the best-matching predicted interval and computes the IoU. This field is the average of those per-target best IoUs.

This metric is useful when a prompt has multiple ground-truth intervals. It asks: "How well was each target segment recovered, on average?"

Example for one record:

- Target best IoUs are `[0.8, 0.2, 0.4]`
- `mean_target_best_iou = (0.8 + 0.2 + 0.4) / 3 = 0.4667`

Then `avg_mean_target_best_iou` is the average of this value across all records in the selected summary scope.

### `avg_mean_predicted_best_iou`
For each predicted interval, the system finds the best-matching target interval and computes the IoU. This field is the average of those per-predicted best IoUs.

This is the mirror of `avg_mean_target_best_iou` and is useful for understanding whether predictions are too broad, too narrow, or fragmented.

Example for one record:

- Predicted best IoUs are `[0.9, 0.5, 0.1, 0.0]`
- `mean_predicted_best_iou = (0.9 + 0.5 + 0.1 + 0.0) / 4 = 0.375`

Then `avg_mean_predicted_best_iou` is the average of this value across records.

### Side-by-side intuition for the two metrics

- `mean_target_best_iou` is target-centric coverage quality.
  It asks: for each true segment, was there at least one good prediction?
- `mean_predicted_best_iou` is prediction-centric precision quality.
  It asks: for each predicted segment, does it correspond well to a true segment?

Typical interpretation patterns:

- High target-centric, low prediction-centric:
  Most true segments are found, but many extra/noisy predicted segments exist.
- Low target-centric, high prediction-centric:
  Predictions that exist are good, but some true segments are missed entirely.
- Both high:
  Good coverage and good precision.
- Both low:
  Poor overlap quality overall.

## Miniature full dataset example for these two metrics

Below is a miniature dataset with three records, shown in a JSONL-like style.

```json
{"sample_key":"r1","target_timestamps":[[0,5],[10,15]],"predicted_timestamps":[[0,1],[2,3],[3,5],[10,12]]}
{"sample_key":"r2","target_timestamps":[[20,30],[40,50]],"predicted_timestamps":[[20,30],[40,45],[60,70]]}
{"sample_key":"r3","target_timestamps":[[100,110]],"predicted_timestamps":[[95,105],[105,115]]}
```

Assume interval duration is `end - start`.

### Record r1

Targets: `[[0,5],[10,15]]`
Predictions: `[[0,1],[2,3],[3,5],[10,12]]`

- Best IoU per target:
  - Target `[0,5]` best is `0.4`
  - Target `[10,15]` best is `0.4`
  - `mean_target_best_iou = (0.4 + 0.4) / 2 = 0.4`

- Best IoU per prediction:
  - `[0,1]` best `0.2`
  - `[2,3]` best `0.2`
  - `[3,5]` best `0.4`
  - `[10,12]` best `0.4`
  - `mean_predicted_best_iou = (0.2 + 0.2 + 0.4 + 0.4) / 4 = 0.3`

### Record r2

Targets: `[[20,30],[40,50]]`
Predictions: `[[20,30],[40,45],[60,70]]`

- Best IoU per target:
  - Target `[20,30]` best `1.0` (perfect match)
  - Target `[40,50]` best `0.5` (overlap 5, union 10)
  - `mean_target_best_iou = (1.0 + 0.5) / 2 = 0.75`

- Best IoU per prediction:
  - `[20,30]` best `1.0`
  - `[40,45]` best `0.5`
  - `[60,70]` best `0.0` (no overlap with any target)
  - `mean_predicted_best_iou = (1.0 + 0.5 + 0.0) / 3 = 0.5`

### Record r3

Targets: `[[100,110]]`
Predictions: `[[95,105],[105,115]]`

- Best IoU per target:
  - Target `[100,110]` has IoU `5/15 = 0.3333` with each prediction, so best is `0.3333`
  - `mean_target_best_iou = 0.3333`

- Best IoU per prediction:
  - `[95,105]` best `0.3333`
  - `[105,115]` best `0.3333`
  - `mean_predicted_best_iou = (0.3333 + 0.3333) / 2 = 0.3333`

  ### Record-level `best_iou`

  For a single record, `best_iou` is the highest IoU among all target-predicted interval pairs.

  In this miniature dataset:

  - r1 pairwise IoUs peak at `0.4`, so `best_iou = 0.4`
  - r2 pairwise IoUs peak at `1.0`, so `best_iou = 1.0`
  - r3 pairwise IoUs peak at `0.3333`, so `best_iou = 0.3333`

  ### Aggregate `avg_best_iou`

  Per-record `best_iou` values:

  - r1: `0.4`
  - r2: `1.0`
  - r3: `0.3333`

  So:

  $$
  avg\_best\_iou = \frac{0.4 + 1.0 + 0.3333}{3} \approx 0.5778
  $$

### Aggregate over the miniature dataset

Per-record `mean_target_best_iou` values:

- r1: `0.4`
- r2: `0.75`
- r3: `0.3333`

So:

$$
avg\_mean\_target\_best\_iou = \frac{0.4 + 0.75 + 0.3333}{3} \approx 0.4944
$$

Per-record `mean_predicted_best_iou` values:

- r1: `0.3`
- r2: `0.5`
- r3: `0.3333`

So:

$$
avg\_mean\_predicted\_best\_iou = \frac{0.3 + 0.5 + 0.3333}{3} \approx 0.3778
$$

For the same miniature dataset, duration-based metrics per record are:

- r1:
  - target total `10`, predicted total `6`, overlap `6`
  - temporal_set_iou `= 6/(10+6-6)=0.6`
  - overlap_over_max `= 6/max(10,6)=0.6`
  - duration_precision `= 6/6=1.0`
  - duration_recall `= 6/10=0.6`
- r2:
  - target total `20`, predicted total `25`, overlap `15`
  - temporal_set_iou `= 15/(20+25-15)=0.5`
  - overlap_over_max `= 15/max(20,25)=0.6`
  - duration_precision `= 15/25=0.6`
  - duration_recall `= 15/20=0.75`
- r3:
  - target total `10`, predicted total `20`, overlap `10`
  - temporal_set_iou `= 10/(10+20-10)=0.5`
  - overlap_over_max `= 10/max(10,20)=0.5`
  - duration_precision `= 10/20=0.5`
  - duration_recall `= 10/10=1.0`

Cross-record averages:

$$
avg\_temporal\_set\_iou = (0.6 + 0.5 + 0.5)/3 \approx 0.5333
$$

$$
avg\_overlap\_over\_max = (0.6 + 0.6 + 0.5)/3 \approx 0.5667
$$

$$
avg\_duration\_precision = (1.0 + 0.6 + 0.5)/3 = 0.7
$$

$$
avg\_duration\_recall = (0.6 + 0.75 + 1.0)/3 \approx 0.7833
$$

Why these differ in this example:

- Target-centric score is higher because most targets have at least one decent match.
- Prediction-centric score is lower because some predictions are weak or spurious (for example `[60,70]` in r2).

### `avg_recall`
The average recall across evaluated records, using the configured `recall_iou_threshold`.

A record gets recall `1.0` when at least one predicted interval overlaps a ground-truth interval with IoU greater than or equal to the threshold.
A record gets recall `0.0` when none of the predictions reach the threshold.

This field answers: "How often did the model successfully retrieve at least one correct segment?"

Example:

- If 8 out of 20 records have recall `1.0` and 12 have recall `0.0`, then:

$$
avg\_recall = \frac{8}{20} = 0.4
$$


## Recall@IoU — Explanation with Edge Cases

### Basic Recall Context

Let’s say:
- 100 real objects in images  
- Model detects 80 objects  
- Out of those, 70 have IoU ≥ 0.5  

Then:
- Recall@0.5 = 70 / 100 = 0.70 (70%)

---

## Important Clarification

Recall is **NOT**:

> number of matching predictions / number of targets

Recall is:

> number of *matched ground-truth objects* / total ground-truth objects

---

## Key Rule (Critical)

In object detection:

> Each ground-truth object can be matched to **at most ONE prediction** (typically the one with the highest IoU).

This is called **one-to-one matching**.

- Extra predictions overlapping the same target are **ignored for recall**
- They count as **false positives** (affect precision, not recall)

---

## Case 1

**Scenario:**
- 1 target interval  
- 5 predicted segments inside it  
- All have IoU ≥ 0.5  

**What happens:**
- Only **1 prediction** is matched to the target  
- The other 4 are duplicates  

**Result:**
- True Positives = 1  
- Total Ground Truth = 1  

Recall@0.5 = 1 / 1 = **1.0**

✅ Recall = **1.0**  
❌ Not 5 / 1  

---

## Case 2

**Scenario:**
- 2 target intervals (A and B)  
- 2 predicted segments  
- Both predictions overlap only target A (IoU ≥ 0.5)  
- Target B is not detected  

**What happens:**
- Only **1 prediction** can match target A  
- The second prediction is a duplicate  
- Target B remains unmatched  

**Result:**
- True Positives = 1  
- Total Ground Truth = 2  

Recall@0.5 = 1 / 2 = **0.5**

✅ Recall = **0.5**  
❌ Not 2 / 2 = 1  

---

## Intuition

Recall answers:

> “How many *unique real objects* did I successfully detect?”

NOT:

> “How many predictions look correct?”

---

## Why This Rule Exists

Without one-to-one matching:
- A model could produce many overlapping predictions for a single object  
- This would artificially inflate recall  

---

## Summary

| Scenario | Recall |
|--------|--------|
| 1 target, 5 overlapping predictions | 1.0 |
| 2 targets, both predictions hit same target | 0.5 |

### `avg_overlap_anywhere_recall`
The average recall across evaluated records using a loose overlap-anywhere rule.

A target interval gets credit even if at least one predicted interval has IoU `> 0.0` with it.

This is the project’s overlap-anywhere test and is separate from the configurable `recall_iou_threshold`.

Example:

- If 8 out of 20 records have overlap-anywhere recall `1.0` and 12 have `0.0`, then:

$$
avg\_overlap\_anywhere\_recall = \frac{8}{20} = 0.4
$$

### `avg_temporal_set_iou`
The average of per-record `temporal_set_iou` values.

Per record:

$$
temporal\_set\_iou = \frac{\text{overlap duration}}{\text{target total duration} + \text{prediction total duration} - \text{overlap duration}}
$$

This is a global interval-set IoU over the full record and helps penalize oversized predictions.

### `avg_overlap_over_max`
The average of per-record `overlap_over_max` values.

Per record:

$$
overlap\_over\_max = \frac{\text{overlap duration}}{\max(\text{target total duration},\text{prediction total duration})}
$$

This is the metric you described: overlap divided by the larger of predicted or target total duration.

### `avg_duration_precision`
The average of per-record duration precision values.

Per record:

$$
duration\_precision = \frac{\text{overlap duration}}{\text{prediction total duration}}
$$

Interpretation: how much predicted duration is actually correct.

### `avg_duration_recall`
The average of per-record duration recall values.

Per record:

$$
duration\_recall = \frac{\text{overlap duration}}{\text{target total duration}}
$$

Interpretation: how much target duration was recovered.

### `total_processing_seconds`
The total wall-clock processing time across all evaluated records included in the summary.

This is useful for estimating compute cost and throughput.

### `avg_processing_seconds`
The average processing time per evaluated record.

This is the throughput view of the same runtime information.

### `total_predicted_total_duration_seconds`
The sum of `predicted_total_duration_seconds` across all evaluated records included in the summary.

This tells you the total merged predicted interval length across the selected report scope.

### `unique_video_count`
The number of distinct videos represented in the summary.

If a run evaluates multiple prompts against the same video, this number can still be `1`.

### `unique_video_duration_seconds_total`
The total duration, in seconds, of the unique videos represented in the summary.

This helps compare evaluation cost against source video length.

### `recall_iou_threshold`
The IoU threshold used to compute Recall@IoU.

For example, if this value is `0.5`, a prediction is counted as correct only when its IoU with a target interval is at least `0.5`.

## Path breakdown fields

The report also groups results by `path_taken`, usually one of:

- `clip`
- `xclip`
- `yolo`
- `audio`
- `ocr`

Each path summary repeats the same fields above, but only for records that used that path.

### `path_counts`
A map from path name to the number of evaluated records routed through that path.

Example:

```json
{
  "clip": 12,
  "xclip": 11,
  "yolo": 11
}
```

## Hyperparameter summary fields

### `hyperparameter_combo_key`
A stable identifier for one specific hyperparameter configuration.

This is a hash-like key used to group results belonging to the same configuration.
It is not itself the configuration, just the identifier for it.

### `hyperparameters`
The actual parameter values used for that combo.

This typically includes values such as:

- `CLIP_THRESHOLD`
- `XCLIP_THRESHOLD`
- `YOLO_MIN_THRESHOLD`
- `YOLO_MAX_USAGE_THRESHOLD`
- `VIDEO_SAMPLING_RATE`
- `MAX_NUMBER_OF_MODIFIED_PROMPTS`
- `FRAME_NEIGHBOUR_RANGE_BEFORE`
- `FRAME_NEIGHBOUR_RANGE_AFTER`

The dashboard shows these values directly so you do not need to inspect the raw JSON by key.

### Combo-level summary fields
Each hyperparameter combo also has its own summary:

- `prompt_count`: how many prompt evaluations used this combo
- `avg_best_iou`: average best IoU for this combo
- `avg_recall`: average recall for this combo
- `avg_processing_seconds`: average runtime for this combo
- `total_processing_seconds`: total runtime for this combo

These are the main fields used to compare one hyperparameter set against another.

## Prompt-row fields in `cumulative_prompt_results.jsonl`

Each line in the JSONL file is one evaluated record. The most important fields are:

- `video`: video filename
- `prompt_id`: prompt label such as `p1`, `p2`, or `p3`
- `prompt`: the text prompt that was evaluated
- `path_taken`: route used to process the prompt
- `target_timestamps`: ground-truth intervals
- `predicted_timestamps`: predicted intervals
- `iou`: nested IoU metrics for that record
- `route_details`: route-specific debug data
- `processing_seconds`: runtime for that record
- `hyperparameters`: the parameter values used for that record
- `hyperparameter_combo_key`: the ID of the parameter combo

## IoU fields

IoU means Intersection over Union.

For two time intervals, it is:

$$
IoU = \frac{\text{intersection duration}}{\text{union duration}}
$$

A value of `1.0` means perfect overlap.
A value of `0.0` means no overlap.

### `best_iou`
The highest IoU obtained between any predicted interval and the target interval(s).

This is the primary score used to judge how well the system localized the relevant segment.

### `mean_target_best_iou`
For each target interval, find the best-matching prediction. Average those values.

This helps when there are multiple ground-truth intervals.

### `mean_predicted_best_iou`
For each predicted interval, find the best-matching target. Average those values.

This helps reveal whether predictions are too noisy or fragmented.

### `pairwise_ious`
The full list of IoU scores between every target interval and every predicted interval.

This is useful for debugging and understanding why the best score was achieved.

### `temporal_set_iou`
Global interval-set IoU for one record using merged intervals:

$$
temporal\_set\_iou = \frac{\text{overlap duration}}{\text{union duration}}
$$

where:

$$
  ext{union duration} = \text{target total duration} + \text{prediction total duration} - \text{overlap duration}
$$

### `overlap_over_max`
Global overlap normalized by the larger side:

$$
overlap\_over\_max = \frac{\text{overlap duration}}{\max(\text{target total duration},\text{prediction total duration})}
$$

### `duration_precision`
Duration-level precision for one record:

$$
duration\_precision = \frac{\text{overlap duration}}{\text{prediction total duration}}
$$

### `duration_recall`
Duration-level recall for one record:

$$
duration\_recall = \frac{\text{overlap duration}}{\text{target total duration}}
$$

### `overlap_duration_seconds`
Total overlap duration between merged target and merged predicted intervals.

### `predicted_total_duration_seconds`
Total duration of merged predicted intervals.

### `target_total_duration_seconds`
Total duration of merged target intervals.

## Recall@IoU

Recall@IoU is a binary success measure computed with an IoU threshold.

Given a threshold like `0.5`:

- If at least one prediction reaches IoU `>= 0.5` with a target interval, the record gets recall `1.0`.
- Otherwise the record gets recall `0.0`.

The project also reports `overlap_anywhere_recall`, which uses a stricter rule:

- If at least one prediction reaches IoU `> 0.0` with a target interval, the record gets overlap-anywhere recall `1.0`.
- Otherwise the record gets overlap-anywhere recall `0.0`.

This metric is stricter than average IoU in a different way:

- `avg_best_iou` tells you how close the best overlap usually is.
- `avg_recall` tells you how often the system crosses the required correctness threshold.

A system can have a moderate average IoU but low recall if it often gets close without crossing the threshold.

## Worked example with your timestamps

Given:

- Targets: `[[0, 5], [10, 15]]`
- Predictions: `[[0, 1], [2, 3], [3, 5], [10, 12]]`

Assuming interval duration is `end - start` and IoU is:

$$
IoU = \frac{\text{overlap duration}}{\text{union duration}}
$$

Pairwise IoUs:

- Target `[0,5]` vs `[0,1]`: $1/5 = 0.2$
- Target `[0,5]` vs `[2,3]`: $1/5 = 0.2$
- Target `[0,5]` vs `[3,5]`: $2/5 = 0.4$
- Target `[0,5]` vs `[10,12]`: $0$
- Target `[10,15]` vs `[0,1]`: $0$
- Target `[10,15]` vs `[2,3]`: $0$
- Target `[10,15]` vs `[3,5]`: $0$
- Target `[10,15]` vs `[10,12]`: $2/5 = 0.4$

So, for this one record:

- `best_iou = 0.4`
- `mean_target_best_iou`:
  - For target `[0,5]`, best is `0.4`
  - For target `[10,15]`, best is `0.4`
  - Mean = `(0.4 + 0.4) / 2 = 0.4`
- `mean_predicted_best_iou`:
  - For prediction `[0,1]`, best is `0.2`
  - For prediction `[2,3]`, best is `0.2`
  - For prediction `[3,5]`, best is `0.4`
  - For prediction `[10,12]`, best is `0.4`
  - Mean = `(0.2 + 0.2 + 0.4 + 0.4) / 4 = 0.3`
- Duration totals:
  - `target_total_duration_seconds = 10`
  - `predicted_total_duration_seconds = 6`
  - `overlap_duration_seconds = 6`
- New duration-overlap metrics:
  - `temporal_set_iou = 6 / (10 + 6 - 6) = 0.6`
  - `overlap_over_max = 6 / max(10, 6) = 0.6`
  - `duration_precision = 6 / 6 = 1.0`
  - `duration_recall = 6 / 10 = 0.6`

Recall@IoU outcomes for this record:

- At threshold `0.5`: recall is `0.0` (no pair reaches `0.5`)
- At threshold `0.4`: recall is `1.0` (there are pairs at `0.4`)

Important note:

- The summary fields prefixed with `avg_` (such as `avg_best_iou`) are averages across many records.
- The values above are per-record values before cross-record averaging.

## Practical reading guide

If you want the shortest interpretation of the report, focus on these fields first:

1. `avg_best_iou` for quality of localization
2. `avg_recall` for thresholded success rate
3. `avg_processing_seconds` for speed
4. `path_counts` for routing behavior
5. `hyperparameter_combo_key` plus `hyperparameters` for configuration comparison

If you want a ranking of hyperparameters, the dashboard's "best avg IoU combo" and combo inspector are the best starting points.
