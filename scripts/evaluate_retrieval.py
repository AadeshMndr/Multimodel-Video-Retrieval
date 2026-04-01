import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, total: int, desc: str = "", unit: str = ""):
            self.total = total
            self.current = 0
            self.desc = desc
            self.unit = unit

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def update(self, n: int = 1):
            self.current += n
            print(f"{self.desc}: {self.current}/{self.total} {self.unit}")

    def tqdm(*args, **kwargs):
        return _TqdmFallback(*args, **kwargs)


# Suppress noisy duplicate Objective-C class logs from bundled FFmpeg dylibs.
# This must be present at process start; re-exec once so the env is guaranteed early.
if os.environ.get("OBJC_PRINT_DUPLICATE_CLASSES") != "NO":
    os.environ["OBJC_PRINT_DUPLICATE_CLASSES"] = "NO"
    os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)

# Suppress Python 3.14 compatibility warning emitted by pydantic.v1 shims.
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@contextmanager
def suppress_stderr_fd() -> Any:
    stderr_fd = sys.stderr.fileno()
    original_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(original_stderr_fd, stderr_fd)
        os.close(original_stderr_fd)
        os.close(devnull_fd)


with suppress_stderr_fd():
    from config import settings
    from router.main_graph import main_workflow
    from router.main_state import Main_State, get_main_state


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
CUMULATIVE_RESULTS_FILENAME = "cumulative_prompt_results.jsonl"
CUMULATIVE_SUMMARY_FILENAME = "cumulative_summary.json"
RUN_STATE_FILENAME = "run_state.json"
RECALL_IOU_THRESHOLD = 0.5


@dataclass
class PromptTask:
    video_path: Path
    prompt: str
    target_timestamps: list[tuple[int, int]]
    prompt_id: str
    label_file: Path
    video_duration_seconds: float | None


@dataclass
class EvaluationTask:
    prompt_task: PromptTask
    hyperparameters: dict[str, Any] | None
    hyperparameter_combo_key: str | None
    hyperparameter_combo_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate video-text retrieval timestamps with IoU scoring."
    )
    parser.add_argument("--videos-dir", default="evaluation/test_videos", help="Directory containing videos.")
    parser.add_argument("--labels-dir", default="evaluation/test_labels", help="Directory containing prompt label JSON files.")
    parser.add_argument(
        "--reports-dir",
        default="evaluation/reports",
        help="Directory where detailed and cumulative reports are stored.",
    )
    parser.add_argument(
        "--label-glob",
        default="*.json",
        help="Glob for label files inside labels directory.",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate output clip files for each prompt evaluation.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Run and append even if sample already exists in cumulative results.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap for number of prompts to evaluate (debug use).",
    )
    parser.add_argument(
        "--recall-iou-threshold",
        type=float,
        default=RECALL_IOU_THRESHOLD,
        help="IoU threshold used by recall@IoU (default: 0.5).",
    )
    parser.add_argument(
        "--enable-random-search-cv",
        action="store_true",
        help="Enable seeded random search over hyperparameters from the search-space JSON.",
    )
    parser.add_argument(
        "--hyperparameter-search-space",
        default="evaluation/hyperparameter_search_space.json",
        help="Path to the hyperparameter search-space JSON file.",
    )
    parser.add_argument(
        "--max-random-combinations",
        type=int,
        default=20,
        help="Maximum number of random hyperparameter combinations to evaluate (default: 20).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible random combination sampling (default: 42).",
    )
    parser.add_argument(
        "--keep-embeddings-between-combinations",
        action="store_true",
        help="Do not clear embeddings between sampled hyperparameter combinations.",
    )
    return parser.parse_args()


def serialize_json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [serialize_json_safe(each) for each in value]
    if isinstance(value, list):
        return [serialize_json_safe(each) for each in value]
    if isinstance(value, dict):
        return {str(key): serialize_json_safe(each) for key, each in value.items()}
    return value


def normalize_hyperparameter_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_hyperparameter_value(each) for each in value]
    if isinstance(value, tuple):
        return [normalize_hyperparameter_value(each) for each in value]
    if isinstance(value, dict):
        return {str(key): normalize_hyperparameter_value(each) for key, each in value.items()}
    return value


def load_hyperparameter_search_space(search_space_path: Path) -> dict[str, list[Any]]:
    payload = json.loads(search_space_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Hyperparameter search-space file must contain a JSON object.")

    normalized: dict[str, list[Any]] = {}
    for name, values in payload.items():
        if not isinstance(name, str) or name.strip() == "":
            raise ValueError("Hyperparameter names must be non-empty strings.")
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Hyperparameter '{name}' must map to a non-empty list.")
        normalized[name] = [normalize_hyperparameter_value(each) for each in values]

    return normalized


def sample_random_hyperparameter_combinations(
    search_space: dict[str, list[Any]],
    max_combinations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_combinations <= 0:
        return []

    parameter_names = sorted(search_space.keys())
    rng = random.Random(seed)
    seen_serialized: set[str] = set()
    sampled: list[dict[str, Any]] = []

    attempt_limit = max(500, max_combinations * 200)
    attempts = 0

    while len(sampled) < max_combinations and attempts < attempt_limit:
        attempts += 1
        candidate = {
            name: normalize_hyperparameter_value(rng.choice(search_space[name])) for name in parameter_names
        }
        serialized = json.dumps(serialize_json_safe(candidate), sort_keys=True)
        if serialized in seen_serialized:
            continue

        seen_serialized.add(serialized)
        sampled.append(candidate)

    return sampled


def hyperparameter_combo_key(hyperparameters: dict[str, Any]) -> str:
    payload = json.dumps(serialize_json_safe(hyperparameters), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def apply_hyperparameters(hyperparameters: dict[str, Any]) -> None:
    for key, value in hyperparameters.items():
        if not hasattr(settings, key):
            raise AttributeError(f"Hyperparameter '{key}' is not defined in config settings.")
        setattr(settings, key, normalize_hyperparameter_value(value))


def clear_embedding_files() -> list[str]:
    removed: list[str] = []
    embedding_paths = [
        Path(f"{settings.EMBEDDING_STORE_FILEPATH}.h5"),
        Path(f"{settings.XCLIP_EMBEDDING_STORE_FILEPATH}.h5"),
    ]

    for embedding_path in embedding_paths:
        if embedding_path.exists():
            embedding_path.unlink()
            removed.append(embedding_path.as_posix())

    return removed


def normalize_timestamps(value: Any) -> list[tuple[int, int]]:
    if value is None:
        return []

    if isinstance(value, list) and len(value) == 2 and all(isinstance(each, (int, float)) for each in value):
        start, end = int(value[0]), int(value[1])
        return [(min(start, end), max(start, end))]

    if isinstance(value, list):
        normalized: list[tuple[int, int]] = []
        for each in value:
            if not (isinstance(each, list) or isinstance(each, tuple)) or len(each) != 2:
                continue
            start, end = int(each[0]), int(each[1])
            normalized.append((min(start, end), max(start, end)))
        return normalized

    return []


def resolve_video_path(videos_dir: Path, label_file: Path, payload: dict[str, Any]) -> Path:
    if "video" in payload and isinstance(payload["video"], str):
        candidate = videos_dir / payload["video"]
        if candidate.exists():
            return candidate

    stem = label_file.stem
    for suffix in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No matching video found for label file '{label_file.name}'. "
        "Either provide 'video' in the label file or ensure same-stem video exists."
    )


def parse_label_file(videos_dir: Path, label_file: Path) -> list[PromptTask]:
    payload = json.loads(label_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Label file '{label_file}' must contain a JSON object.")

    video_path = resolve_video_path(videos_dir, label_file, payload)

    video_duration_seconds = read_video_duration_seconds(video_path)

    prompts = payload.get("prompts", [])
    if not isinstance(prompts, list):
        raise ValueError(f"Label file '{label_file}' has invalid 'prompts'; expected a list.")

    tasks: list[PromptTask] = []
    for index, each_prompt in enumerate(prompts):
        if not isinstance(each_prompt, dict):
            continue

        prompt_text = each_prompt.get("prompt")
        if not isinstance(prompt_text, str) or prompt_text.strip() == "":
            continue

        target_timestamps = normalize_timestamps(
            each_prompt.get("target_timestamps", each_prompt.get("target_timestamp"))
        )

        task = PromptTask(
            video_path=video_path,
            prompt=prompt_text.strip(),
            target_timestamps=target_timestamps,
            prompt_id=str(each_prompt.get("id", f"{label_file.stem}_{index}")),
            label_file=label_file,
            video_duration_seconds=video_duration_seconds,
        )
        tasks.append(task)

    return tasks


def interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    first_start, first_end = first
    second_start, second_end = second

    intersection = max(0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)

    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def read_video_duration_seconds(video_path: Path) -> float | None:
    try:
        import cv2  # type: ignore
    except Exception:  # noqa: BLE001
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None

    try:
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count > 0.0 and fps > 0.0:
            duration = frame_count / fps
            return round(duration, 3)
    finally:
        capture.release()

    return None


def compare_timestamps(
    predicted: list[tuple[int, int]],
    target: list[tuple[int, int]],
    recall_iou_threshold: float,
) -> dict[str, Any]:
    if not predicted and not target:
        return {
            "best_iou": 1.0,
            "mean_target_best_iou": 1.0,
            "mean_predicted_best_iou": 1.0,
            "recall": 1.0,
            "recall_iou_threshold": recall_iou_threshold,
            "pairwise_ious": [],
        }

    if not predicted or not target:
        return {
            "best_iou": 0.0,
            "mean_target_best_iou": 0.0,
            "mean_predicted_best_iou": 0.0,
            "recall": 0.0,
            "recall_iou_threshold": recall_iou_threshold,
            "pairwise_ious": [],
        }

    pairwise = []
    for target_range in target:
        for predicted_range in predicted:
            score = interval_iou(target_range, predicted_range)
            pairwise.append(
                {
                    "target": list(target_range),
                    "predicted": list(predicted_range),
                    "iou": score,
                }
            )

    target_bests: list[float] = []
    for target_range in target:
        best = max(interval_iou(target_range, each_predicted) for each_predicted in predicted)
        target_bests.append(best)

    predicted_bests: list[float] = []
    for predicted_range in predicted:
        best = max(interval_iou(predicted_range, each_target) for each_target in target)
        predicted_bests.append(best)

    recall_hits = sum(1 for each_best in target_bests if each_best >= recall_iou_threshold)
    recall = recall_hits / len(target_bests)

    return {
        "best_iou": max((each["iou"] for each in pairwise), default=0.0),
        "mean_target_best_iou": sum(target_bests) / len(target_bests),
        "mean_predicted_best_iou": sum(predicted_bests) / len(predicted_bests),
        "recall": recall,
        "recall_iou_threshold": recall_iou_threshold,
        "pairwise_ious": pairwise,
    }


def sample_key(task: PromptTask, hyperparameter_combo_key_value: str | None = None) -> str:
    normalized_targets = sorted(task.target_timestamps)
    payload = f"{task.video_path.as_posix()}|{task.prompt}|{normalized_targets}|{hyperparameter_combo_key_value or 'default'}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_existing_keys(cumulative_jsonl: Path) -> set[str]:
    if not cumulative_jsonl.exists():
        return set()

    keys: set[str] = set()
    with cumulative_jsonl.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = payload.get("sample_key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def summarize_results(records: list[dict[str, Any]], recall_iou_threshold: float) -> dict[str, Any]:
    if not records:
        return {
            "prompt_count": 0,
            "avg_best_iou": 0.0,
            "avg_mean_target_best_iou": 0.0,
            "avg_mean_predicted_best_iou": 0.0,
            "avg_recall": 0.0,
            "total_processing_seconds": 0.0,
            "avg_processing_seconds": 0.0,
            "unique_video_count": 0,
            "unique_video_duration_seconds_total": 0.0,
            "path_counts": {},
        }

    avg_best_iou = sum(each["iou"]["best_iou"] for each in records) / len(records)
    avg_target = sum(each["iou"]["mean_target_best_iou"] for each in records) / len(records)
    avg_predicted = sum(each["iou"]["mean_predicted_best_iou"] for each in records) / len(records)
    avg_recall = sum(each["iou"]["recall"] for each in records) / len(records)
    total_processing_seconds = sum(float(each.get("processing_seconds", 0.0) or 0.0) for each in records)
    avg_processing_seconds = total_processing_seconds / len(records)

    video_durations: dict[str, float] = {}
    for each in records:
        video_path = str(each.get("video_path", ""))
        duration = each.get("video_duration_seconds")
        if video_path and isinstance(duration, (int, float)):
            video_durations[video_path] = float(duration)

    path_counts: dict[str, int] = defaultdict(int)
    for each in records:
        path_counts[str(each.get("path_taken", "unknown"))] += 1

    return {
        "prompt_count": len(records),
        "avg_best_iou": avg_best_iou,
        "avg_mean_target_best_iou": avg_target,
        "avg_mean_predicted_best_iou": avg_predicted,
        "avg_recall": avg_recall,
        "total_processing_seconds": total_processing_seconds,
        "avg_processing_seconds": avg_processing_seconds,
        "unique_video_count": len(video_durations),
        "unique_video_duration_seconds_total": sum(video_durations.values()),
        "recall_iou_threshold": recall_iou_threshold,
        "path_counts": dict(path_counts),
    }


def summarize_by_hyperparameter_combo(records: list[dict[str, Any]], recall_iou_threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    combo_payloads: dict[str, dict[str, Any]] = {}

    for record in records:
        combo_key = str(record.get("hyperparameter_combo_key", "default"))
        grouped[combo_key].append(record)
        combo_payload = record.get("hyperparameters")
        if isinstance(combo_payload, dict):
            combo_payloads[combo_key] = combo_payload

    summaries: list[dict[str, Any]] = []
    for combo_key in sorted(grouped.keys()):
        combo_records = grouped[combo_key]
        summaries.append(
            {
                "hyperparameter_combo_key": combo_key,
                "hyperparameters": combo_payloads.get(combo_key),
                "summary": summarize_results(combo_records, recall_iou_threshold),
            }
        )

    return summaries


def summarize_by_path(records: list[dict[str, Any]], recall_iou_threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        path_name = str(record.get("path_taken", "unknown"))
        grouped[path_name].append(record)

    summaries: list[dict[str, Any]] = []
    for path_name in sorted(grouped.keys()):
        path_records = grouped[path_name]
        summaries.append(
            {
                "path_taken": path_name,
                "summary": summarize_results(path_records, recall_iou_threshold),
            }
        )
    return summaries


def write_per_video_reports(
    per_video_records: dict[str, list[dict[str, Any]]],
    run_dir: Path,
    recall_iou_threshold: float,
) -> None:
    per_video_dir = run_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    for video_name, records in per_video_records.items():
        video_duration_seconds = None
        if records:
            duration = records[0].get("video_duration_seconds")
            if isinstance(duration, (int, float)):
                video_duration_seconds = float(duration)

        payload = {
            "video": video_name,
            "video_duration_seconds": video_duration_seconds,
            "summary": summarize_results(records, recall_iou_threshold),
            "summary_by_path": summarize_by_path(records, recall_iou_threshold),
            "summary_by_hyperparameter_combo": summarize_by_hyperparameter_combo(records, recall_iou_threshold),
            "prompt_results": records,
        }
        report_path = per_video_dir / f"{Path(video_name).stem}.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_cumulative_summary(
    reports_dir: Path,
    all_existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    skipped_existing: int,
    recall_iou_threshold: float,
) -> None:
    payload = {
        "last_updated_utc": datetime.now(UTC).isoformat(),
        "cumulative": summarize_results(all_existing_records, recall_iou_threshold),
        "cumulative_by_path": summarize_by_path(all_existing_records, recall_iou_threshold),
        "cumulative_by_hyperparameter_combo": summarize_by_hyperparameter_combo(
            all_existing_records,
            recall_iou_threshold,
        ),
        "last_run": {
            "new_prompt_count": len(new_records),
            "skipped_existing_count": skipped_existing,
            "summary": summarize_results(new_records, recall_iou_threshold),
            "summary_by_path": summarize_by_path(new_records, recall_iou_threshold),
            "summary_by_hyperparameter_combo": summarize_by_hyperparameter_combo(
                new_records,
                recall_iou_threshold,
            ),
        },
    }

    summary_path = reports_dir / CUMULATIVE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_task(
    task: PromptTask,
    reports_dir: Path,
    generate_output_video: bool,
    recall_iou_threshold: float,
    hyperparameters: dict[str, Any] | None = None,
    hyperparameter_combo_key_value: str | None = None,
    hyperparameter_combo_index: int | None = None,
) -> dict[str, Any]:
    output_filename = f"eval_{task.video_path.stem}_{hashlib.md5(task.prompt.encode('utf-8')).hexdigest()[:10]}.mp4"
    output_path = str(reports_dir / "generated_clips" / output_filename)

    state = get_main_state(
        video_path=str(task.video_path),
        user_text=task.prompt,
        output_path=output_path,
        generate_output_video=generate_output_video,
    )

    started_monotonic = time.perf_counter()
    try:
        final_state: Main_State = main_workflow.invoke(state)  # type: ignore
    finally:
        state["embedding_store"].close()
    processing_seconds = time.perf_counter() - started_monotonic

    predicted_timestamps = [
        (int(start), int(end)) for start, end in final_state.get("timestamps", [])
    ]

    iou_scores = compare_timestamps(
        predicted_timestamps,
        task.target_timestamps,
        recall_iou_threshold,
    )
    route_details = dict(final_state.get("route_details", {}))
    route_details.pop("frames_scores", None)

    return {
        "sample_key": sample_key(task, hyperparameter_combo_key_value),
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "video": task.video_path.name,
        "video_path": task.video_path.as_posix(),
        "video_duration_seconds": task.video_duration_seconds,
        "processing_seconds": processing_seconds,
        "label_file": task.label_file.as_posix(),
        "prompt_id": task.prompt_id,
        "prompt": task.prompt,
        "path_taken": final_state.get("logical_path_choosen", "unknown"),
        "target_timestamps": [list(each) for each in task.target_timestamps],
        "predicted_timestamps": [list(each) for each in predicted_timestamps],
        "iou": iou_scores,
        "route_details": route_details,
        "matched_frames_count": len(final_state.get("matched_frames", [])),
        "hyperparameters": serialize_json_safe(hyperparameters),
        "hyperparameter_combo_key": hyperparameter_combo_key_value,
        "hyperparameter_combo_index": hyperparameter_combo_index,
    }


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.recall_iou_threshold <= 1.0:
        raise ValueError("--recall-iou-threshold must be between 0.0 and 1.0.")

    videos_dir = Path(args.videos_dir)
    labels_dir = Path(args.labels_dir)
    reports_dir = Path(args.reports_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "generated_clips").mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob(args.label_glob))
    if not label_files:
        raise FileNotFoundError(f"No label files found in {labels_dir} using glob '{args.label_glob}'.")

    all_tasks: list[PromptTask] = []
    for label_file in label_files:
        all_tasks.extend(parse_label_file(videos_dir, label_file))

    if args.max_prompts is not None:
        all_tasks = all_tasks[: max(0, args.max_prompts)]

    selected_combinations: list[dict[str, Any]] = []
    if args.enable_random_search_cv:
        search_space_path = Path(args.hyperparameter_search_space)
        if not search_space_path.exists():
            raise FileNotFoundError(f"Hyperparameter search-space file does not exist: {search_space_path}")
        search_space = load_hyperparameter_search_space(search_space_path)
        selected_combinations = sample_random_hyperparameter_combinations(
            search_space=search_space,
            max_combinations=args.max_random_combinations,
            seed=args.random_seed,
        )
        if not selected_combinations:
            raise ValueError("No hyperparameter combinations were sampled. Increase max-random-combinations.")
    else:
        selected_combinations = [{}]

    cumulative_jsonl = reports_dir / CUMULATIVE_RESULTS_FILENAME
    existing_keys = set() if args.force_recompute else load_existing_keys(cumulative_jsonl)

    pending_tasks: list[EvaluationTask] = []
    skipped_existing = 0

    for combo_index, hyperparameters in enumerate(selected_combinations, start=1):
        combo_key_value = hyperparameter_combo_key(hyperparameters) if args.enable_random_search_cv else None

        for task in all_tasks:
            key = sample_key(task, combo_key_value)
            if key in existing_keys:
                skipped_existing += 1
                continue

            pending_tasks.append(
                EvaluationTask(
                    prompt_task=task,
                    hyperparameters=hyperparameters if args.enable_random_search_cv else None,
                    hyperparameter_combo_key=combo_key_value,
                    hyperparameter_combo_index=combo_index if args.enable_random_search_cv else None,
                )
            )

    if not pending_tasks:
        print("No new prompts to evaluate. All inputs are already present in cumulative report.")
        if cumulative_jsonl.exists():
            all_records = [
                json.loads(each)
                for each in cumulative_jsonl.read_text(encoding="utf-8").splitlines()
                if each.strip()
            ]
            upsert_cumulative_summary(
                reports_dir,
                all_records,
                [],
                skipped_existing,
                args.recall_iou_threshold,
            )
        return

    pending_tasks.sort(
        key=lambda each: (
            each.hyperparameter_combo_index or 0,
            each.prompt_task.video_path.as_posix(),
            each.prompt_task.prompt_id,
        )
    )

    print(
        "Resume status: "
        f"planned={len(all_tasks) * len(selected_combinations)}, "
        f"already_done={skipped_existing}, "
        f"remaining_now={len(pending_tasks)}"
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = reports_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = run_dir / RUN_STATE_FILENAME

    selected_combinations_payload: dict[str, Any] | None = None
    if args.enable_random_search_cv:
        selected_combinations_payload = {
            "random_seed": args.random_seed,
            "max_random_combinations": args.max_random_combinations,
            "selected_combinations_count": len(selected_combinations),
            "selected_combinations": [
                {
                    "hyperparameter_combo_index": index,
                    "hyperparameter_combo_key": hyperparameter_combo_key(combo),
                    "hyperparameters": serialize_json_safe(combo),
                }
                for index, combo in enumerate(selected_combinations, start=1)
            ],
        }
        write_json(run_dir / "selected_hyperparameter_combinations.json", selected_combinations_payload)

    write_json(
        run_state_path,
        {
            "run_id": run_id,
            "status": "running",
            "started_utc": datetime.now(UTC).isoformat(),
            "completed_tasks": 0,
            "total_tasks": len(pending_tasks),
            "skipped_existing": skipped_existing,
            "last_completed_sample_key": None,
            "last_error": None,
        },
    )

    run_records: list[dict[str, Any]] = []
    per_video_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    last_combo_key: str | None = None
    embedding_clears: list[dict[str, Any]] = []
    interrupted = False
    last_error: dict[str, Any] | None = None

    with cumulative_jsonl.open("a", encoding="utf-8") as cumulative_append_file:
        with tqdm(total=len(pending_tasks), desc="Evaluating prompts", unit="prompt") as progress_bar:
            for evaluation_task in pending_tasks:
                combo_key_value = evaluation_task.hyperparameter_combo_key
                should_switch_combo = combo_key_value != last_combo_key

                try:
                    if should_switch_combo:
                        if evaluation_task.hyperparameters is not None:
                            apply_hyperparameters(evaluation_task.hyperparameters)

                        if not args.keep_embeddings_between_combinations:
                            removed_paths = clear_embedding_files()
                            embedding_clears.append(
                                {
                                    "hyperparameter_combo_index": evaluation_task.hyperparameter_combo_index,
                                    "hyperparameter_combo_key": combo_key_value,
                                    "removed_files": removed_paths,
                                }
                            )

                        last_combo_key = combo_key_value

                    record = evaluate_task(
                        evaluation_task.prompt_task,
                        reports_dir,
                        generate_output_video=args.generate_video,
                        recall_iou_threshold=args.recall_iou_threshold,
                        hyperparameters=evaluation_task.hyperparameters,
                        hyperparameter_combo_key_value=combo_key_value,
                        hyperparameter_combo_index=evaluation_task.hyperparameter_combo_index,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    last_error = {
                        "kind": "KeyboardInterrupt",
                        "message": "Evaluation interrupted by user.",
                        "at_task": {
                            "video": evaluation_task.prompt_task.video_path.name,
                            "prompt_id": evaluation_task.prompt_task.prompt_id,
                            "hyperparameter_combo_index": evaluation_task.hyperparameter_combo_index,
                            "hyperparameter_combo_key": combo_key_value,
                        },
                    }
                    break
                except Exception as error:  # noqa: BLE001
                    last_error = {
                        "kind": type(error).__name__,
                        "message": str(error),
                        "at_task": {
                            "video": evaluation_task.prompt_task.video_path.name,
                            "prompt_id": evaluation_task.prompt_task.prompt_id,
                            "hyperparameter_combo_index": evaluation_task.hyperparameter_combo_index,
                            "hyperparameter_combo_key": combo_key_value,
                        },
                        "traceback": traceback.format_exc(),
                    }
                    break

                run_records.append(record)
                per_video_records[evaluation_task.prompt_task.video_path.name].append(record)
                existing_keys.add(record["sample_key"])

                # Persist each completed prompt immediately so interrupted runs can resume safely.
                cumulative_append_file.write(json.dumps(record) + "\n")
                cumulative_append_file.flush()

                progress_bar.update(1)

                write_json(
                    run_state_path,
                    {
                        "run_id": run_id,
                        "status": "running",
                        "started_utc": run_id,
                        "updated_utc": datetime.now(UTC).isoformat(),
                        "completed_tasks": len(run_records),
                        "total_tasks": len(pending_tasks),
                        "skipped_existing": skipped_existing,
                        "last_completed_sample_key": record["sample_key"],
                        "last_error": None,
                    },
                )

    write_per_video_reports(per_video_records, run_dir, args.recall_iou_threshold)

    if embedding_clears:
        (run_dir / "embedding_clears.json").write_text(
            json.dumps(embedding_clears, indent=2),
            encoding="utf-8",
        )

    run_summary_payload = {
        "run_id": run_id,
        "started_utc": run_id,
        "finished_utc": datetime.now(UTC).isoformat(),
        "status": "interrupted" if interrupted else ("failed" if last_error else "completed"),
        "last_error": last_error,
        "settings": {
            "videos_dir": videos_dir.as_posix(),
            "labels_dir": labels_dir.as_posix(),
            "generate_video": bool(args.generate_video),
            "force_recompute": bool(args.force_recompute),
            "max_prompts": args.max_prompts,
            "recall_iou_threshold": args.recall_iou_threshold,
            "enable_random_search_cv": bool(args.enable_random_search_cv),
            "hyperparameter_search_space": args.hyperparameter_search_space,
            "max_random_combinations": args.max_random_combinations,
            "random_seed": args.random_seed,
            "keep_embeddings_between_combinations": bool(args.keep_embeddings_between_combinations),
        },
        "counts": {
            "total_label_tasks": len(all_tasks),
            "selected_combinations": len(selected_combinations),
            "planned_evaluations": len(all_tasks) * len(selected_combinations),
            "evaluated_new": len(run_records),
            "skipped_existing": skipped_existing,
        },
        "summary": summarize_results(run_records, args.recall_iou_threshold),
        "summary_by_path": summarize_by_path(run_records, args.recall_iou_threshold),
        "summary_by_hyperparameter_combo": summarize_by_hyperparameter_combo(
            run_records,
            args.recall_iou_threshold,
        ),
    }
    write_json(run_dir / "run_summary.json", run_summary_payload)

    write_json(
        run_state_path,
        {
            "run_id": run_id,
            "status": "interrupted" if interrupted else ("failed" if last_error else "completed"),
            "started_utc": run_id,
            "finished_utc": datetime.now(UTC).isoformat(),
            "completed_tasks": len(run_records),
            "total_tasks": len(pending_tasks),
            "skipped_existing": skipped_existing,
            "last_completed_sample_key": run_records[-1]["sample_key"] if run_records else None,
            "last_error": last_error,
        },
    )

    all_records = [
        json.loads(each)
        for each in cumulative_jsonl.read_text(encoding="utf-8").splitlines()
        if each.strip()
    ]
    upsert_cumulative_summary(
        reports_dir,
        all_records,
        run_records,
        skipped_existing,
        args.recall_iou_threshold,
    )

    if interrupted:
        print(
            "Evaluation interrupted by user. "
            f"Completed before stop: {len(run_records)}. "
            "Rerun the same command to resume from remaining items. "
            f"Run report: {run_dir.as_posix()}"
        )
    elif last_error:
        print(
            "Evaluation stopped due to error. "
            f"Completed before error: {len(run_records)}. "
            "Already completed items were saved; rerun the same command to continue. "
            f"Run report: {run_dir.as_posix()}"
        )
    else:
        print(
            f"Evaluation complete. New records: {len(run_records)}, skipped existing: {skipped_existing}. "
            f"Run report: {run_dir.as_posix()}"
        )


if __name__ == "__main__":
    main()
