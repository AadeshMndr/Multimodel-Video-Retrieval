# XCLIP Path: Detailed Flow

This document explains exactly what happens when the router selects the `xclip` path.

## 1) How `xclip` gets selected

- Routing begins in `prompt_analyzer`.
- The LLM analyzer chooses one of: `clip`, `xclip`, `yolo`, `audio`.
- If `xclip` is chosen, graph execution goes to `xclip_logic`.

Relevant files:
- `service_layer/llm_service/node_functions.py`
- `router/main_logic.py`
- `router/main_graph.py`

## 2) Preprocessing before route node execution

All non-audio routes run the preprocessing workflow first:

1. `sample_frames`
2. `remove_similar_frames`
3. `batch_frames`

This happens in:
- `service_layer/video_service/graph.py` (`pre_workflow`)
- `router/main_logic.py` (`preprocess`)

### Important detail for XCLIP

Even though preprocessing creates all three factories, the current XCLIP route uses:
- `sampled_generator_factory`

It does **not** use:
- `reduced_generator_factory`
- `batched_generator_factory`

So XCLIP currently receives sampled frames (after frame sampling), but not the deduplicated frame stream and not prebuilt frame batches.

## 3) How frames are received in `xclip_logic`

Inside `router/logic_routes/xclip_logic.py`:

- Router does **not** materialize frame lists anymore.
- Router passes `sampled_generator_factory` into `xclip_service` state.
- `xclip_service` calls `XCLIP_Processor.find_temporal_matches(...)` with that factory.

The frame factory yields items shaped as:
- `(start_frame, end_frame, pil_image)`

For sampled data, `start_frame == end_frame` for each element.

This keeps the router thin and moves frame handling to service/infrastructure layers.

## 4) What XCLIP does with those frames

Implementation file:
- `infrastructure/xclip_processor.py`

### Step-by-step

1. Build prompt list
   - Uses modified prompts from the LLM prompt-variation workflow.

2. Encode text prompts
   - Processor creates text embeddings for all prompts.
   - Embeddings are normalized.

3. Load cached window embeddings if available
   - Route creates an `Embedding_Store` specifically for XCLIP.
   - Cache key includes video path and window/sampling settings.
   - If cache exists, processor loads:
     - precomputed window embeddings
     - corresponding `(start_frame, end_frame)` ranges
   - In this case, frame generator is not consumed.

4. If cache is missing, generate and store video window embeddings
   - Processor consumes the sampled frame generator into a list.
   - Uses settings:
     - `VIDEO_SAMPLING_RATE`
     - `XCLIP_WINDOW_SECONDS`
     - `XCLIP_STEP_SECONDS`
     - `XCLIP_FRAMES_PER_CLIP`
   - Slides a temporal window over sampled frames.
   - Uniformly picks `XCLIP_FRAMES_PER_CLIP` frames inside each window.
   - Encodes each selected sequence with `XCLIPModel.get_video_features(...)`.
   - Stores all window embeddings + frame ranges into H5 for reuse.

5. Compute similarity against text embeddings
   - Similarity matrix: `window_embeddings @ text_embeddings.T`.
   - Per-window score = max across prompt variants.
   - Window is kept if score >= `XCLIP_THRESHOLD`.

6. Merge nearby/overlapping matched windows
   - Uses `XCLIP_MERGE_GAP_SECONDS` converted to frame gap.
   - Produces final `matched_frames` ranges.

7. Return stats
   - `window_count`, `matched_window_count`, `mean`, `median`, `max`, `min`, `cache_hit`.

## 5) Mac compatibility behavior

In `XCLIP_Processor`:

- Device is resolved from config (`cuda`/`mps`/`cpu` fallback).
- If `mps` is selected, `PYTORCH_ENABLE_MPS_FALLBACK=1` is set.
- If an MPS operation fails, processor falls back to CPU and retries.

## 6) What happens after `xclip_logic` returns

`xclip_logic` returns:
- `matched_frames`
- `route_details` with path and score stats

Then graph moves to `postprocess` (`parallel_post_process`):

1. Put `matched_frame_range` into `video_state`.
2. Run `timestamp_workflow`:
   - `expand_frame_range` (adds neighbor frames, merges continuous spans)
   - `get_timestamps` (frame ranges -> second ranges)
3. If `generate_output_video=True`:
   - Start background thread to render output clip video.
4. Stream timestamps immediately; finish response after background video completes.

Relevant files:
- `router/main_logic.py`
- `service_layer/video_service/graph.py`
- `service_layer/video_service/node_functions.py`
- `infrastructure/video_processor.py`

## 7) Settings that control XCLIP behavior

Defined in `config.py`:

### `XCLIP_MODEL_NAME`
- What it does: chooses which XCLIP checkpoint is loaded.
- Current value: `microsoft/xclip-base-patch16`.
- Example: switching to a smaller checkpoint can reduce memory/latency but may reduce retrieval quality.

### `XCLIP_THRESHOLD`
- What it does: minimum similarity score for keeping a window.
- Current value: `0.250`.
- Example:
   - `0.20` => more recall (more windows kept, more false positives)
   - `0.35` => more precision (fewer windows kept, possible misses)

### `XCLIP_WINDOW_SECONDS`
- What it does: temporal span of each sliding window before selecting frames.
- Current value: `8.0` seconds.
- Example: at sampled 2 FPS, an 8-second window covers ~16 sampled frames before down-selecting to `XCLIP_FRAMES_PER_CLIP`.

### `XCLIP_STEP_SECONDS`
- What it does: how far the sliding window moves each iteration.
- Current value: `2.0` seconds.
- Example:
   - `2.0` with `8.0` window => high overlap (better temporal coverage, higher compute)
   - `8.0` with `8.0` window => no overlap (faster, may miss transitions)

### `XCLIP_FRAMES_PER_CLIP`
- What it does: exact number of frames selected per window for model input.
- Current value: `8`.
- Example: if a window has 16 sampled frames, 8 are selected uniformly across the full span to preserve temporal coverage.

### `XCLIP_MERGE_GAP_SECONDS`
- What it does: merges adjacent matched windows when they are close in time.
- Current value: `2.0` seconds.
- Example: two matches at `10-14s` and `15-20s` get merged if gap <= merge threshold.

### `ENABLE_XCLIP_EMBEDDING_STORAGE`
- What it does: enables H5 caching for XCLIP window embeddings.
- Current value: `True`.
- Example: first query computes/stores window embeddings; later prompts for same video+window settings reuse them.

### `XCLIP_EMBEDDING_STORE_FILEPATH`
- What it does: H5 filename prefix for XCLIP cache.
- Current value: `xclip_embeddings` (stored as `xclip_embeddings.h5`).
- Example: change this to isolate experiments from production cache.

### `XCLIP_EMBEDDING_DIMENSION`
- What it does: embedding width expected by the XCLIP cache dataset.
- Current value: `512`.
- Example: must match model output dimension; wrong value can cause shape mismatch when writing cache.

### `XCLIP_EMBEDDING_CHUNK_SIZE`
- What it does: H5 dataset chunk size for cached embeddings.
- Current value: `256`.
- Example:
   - larger chunks improve sequential read throughput,
   - smaller chunks can reduce waste for tiny videos.

### `VIDEO_SAMPLING_RATE` (shared video setting)
- What it does: frame downsampling before XCLIP windowing.
- Current value: `15` (keep every 15th frame).
- Example: on a 30 FPS video, sampled stream is ~2 FPS, which directly affects windows and compute cost.

## 8) Practical implications of current implementation

- XCLIP path is temporal-window based over sampled frames.
- It reuses cached window embeddings when available, independent of the text prompt.
- On cache miss, it loads sampled frames and computes window embeddings once, then persists them.
- It does not currently use the reduced-frame generator or CLIP batch generator.
- Output contract remains consistent with other visual paths (`matched_frames` -> timestamp refinement -> optional video output).

## 9) Full worked example (30 FPS, 60s video, target event at 5s-10s)

Assume:
- Video FPS = `30`
- Video duration = `60s`
- Total frames = `1800`
- The scene you want is from `5s` to `10s` (original frame range `150` to `300`)

And config values:
- `VIDEO_SAMPLING_RATE = 15`
- `XCLIP_WINDOW_SECONDS = 8.0`
- `XCLIP_STEP_SECONDS = 2.0`
- `XCLIP_FRAMES_PER_CLIP = 8`
- `XCLIP_THRESHOLD = 0.25`
- `XCLIP_MERGE_GAP_SECONDS = 2.0`
- `FRAME_NEIGHBOUR_RANGE_BEFORE = 30`
- `FRAME_NEIGHBOUR_RANGE_AFTER = 30`

### A) Preprocess: sampled frame stream

`VIDEO_SAMPLING_RATE = 15` means keep every 15th frame.

- Effective sampled FPS = `30 / 15 = 2 FPS`
- Sampled timestamps are every `0.5s`
- Number of sampled frames in 60s = `1800 / 15 = 120`

Target event (`5s` to `10s`) is represented in sampled stream around:
- `5.0, 5.5, 6.0, ... , 10.0` seconds

### B) Window construction for XCLIP

From sampled FPS = `2`:

- Frames per window = `int(8.0 * 2) = 16`
- Step size in sampled frames = `int(2.0 * 2) = 4`

So windows slide every 2 seconds with strong overlap.

Examples of window time spans:
- Window 1: `0.0s -> 7.5s`
- Window 2: `2.0s -> 9.5s`
- Window 3: `4.0s -> 11.5s`
- Window 4: `6.0s -> 13.5s`

Your target segment (`5s-10s`) is best covered by windows like `2.0-9.5`, `4.0-11.5`, and `6.0-13.5`.

### C) Frame selection per window

Each 16-frame window is down-selected to exactly `8` uniformly spaced frames (`XCLIP_FRAMES_PER_CLIP = 8`).

Meaning: XCLIP sees a compact temporal summary of each 8-second span, instead of one static frame.

### D) Embedding + similarity check

For each window:
1. Build one video embedding (shape width = `XCLIP_EMBEDDING_DIMENSION`, currently `512`).
2. Compare against text embedding(s) from prompt variants.
3. Take max similarity score for that window.
4. Keep window if score `>= 0.25` (`XCLIP_THRESHOLD`).

Example outcome:
- `2.0-9.5s` score `0.29` -> keep
- `4.0-11.5s` score `0.41` -> keep
- `6.0-13.5s` score `0.31` -> keep
- Most unrelated windows score `< 0.25` -> reject

### E) Merge matched windows

Matched windows are represented in original frame numbers and then merged.

With overlap in the example above, they merge into one continuous range roughly around:
- about `4s` to `13.5s` before postprocess expansion

`XCLIP_MERGE_GAP_SECONDS = 2.0` also merges near-adjacent matches that are close but not strictly overlapping.

### F) Postprocess expansion + timestamp conversion

Then shared postprocess runs:
- Expand each side by 30 frames (`FRAME_NEIGHBOUR_RANGE_BEFORE/AFTER`)
- At 30 FPS, 30 frames = `1s`

So an example merged range near `4s-13.5s` becomes roughly `3s-14.5s` in frames.

Timestamp conversion uses integer seconds, so you may see output like:
- `(3, 14)`

This is expected: the system intentionally returns a buffered clip around the detected event for safer recall.

### G) Where caching helps (prompt-independent video embeddings)

First prompt on that video/settings:
- XCLIP computes all window embeddings and stores them in `xclip_embeddings.h5`.

Second prompt on the same video/settings:
- XCLIP loads stored window embeddings.
- Only text embeddings are recomputed.
- Similarity scoring runs again (fast), without recomputing video embeddings.

Cache identity includes video path + sampling/window config, so if you change values like
`VIDEO_SAMPLING_RATE`, `XCLIP_WINDOW_SECONDS`, `XCLIP_STEP_SECONDS`, or `XCLIP_FRAMES_PER_CLIP`,
a new cache partition is used.
