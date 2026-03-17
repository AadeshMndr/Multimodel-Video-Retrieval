import argparse
import hashlib
import json
import sys
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from router.main_graph import main_workflow
from router.main_state import Main_State, get_main_state


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
CUMULATIVE_RESULTS_FILENAME = "cumulative_prompt_results.jsonl"
CUMULATIVE_SUMMARY_FILENAME = "cumulative_summary.json"


@dataclass
class PromptTask:
    video_path: Path
    prompt: str
    target_timestamps: list[tuple[int, int]]
    prompt_id: str
    label_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate video-text retrieval timestamps with IoU scoring."
    )
    parser.add_argument("--videos-dir", required=True, help="Directory containing videos.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing prompt label JSON files.")
    parser.add_argument(
        "--reports-dir",
        default="outputs/evaluation",
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
    return parser.parse_args()


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


def compare_timestamps(
    predicted: list[tuple[int, int]],
    target: list[tuple[int, int]],
) -> dict[str, Any]:
    if not predicted and not target:
        return {
            "best_iou": 1.0,
            "mean_target_best_iou": 1.0,
            "mean_predicted_best_iou": 1.0,
            "pairwise_ious": [],
        }

    if not predicted or not target:
        return {
            "best_iou": 0.0,
            "mean_target_best_iou": 0.0,
            "mean_predicted_best_iou": 0.0,
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

    return {
        "best_iou": max((each["iou"] for each in pairwise), default=0.0),
        "mean_target_best_iou": sum(target_bests) / len(target_bests),
        "mean_predicted_best_iou": sum(predicted_bests) / len(predicted_bests),
        "pairwise_ious": pairwise,
    }


def sample_key(task: PromptTask) -> str:
    normalized_targets = sorted(task.target_timestamps)
    payload = f"{task.video_path.as_posix()}|{task.prompt}|{normalized_targets}"
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


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "prompt_count": 0,
            "avg_best_iou": 0.0,
            "avg_mean_target_best_iou": 0.0,
            "avg_mean_predicted_best_iou": 0.0,
            "path_counts": {},
        }

    avg_best_iou = sum(each["iou"]["best_iou"] for each in records) / len(records)
    avg_target = sum(each["iou"]["mean_target_best_iou"] for each in records) / len(records)
    avg_predicted = sum(each["iou"]["mean_predicted_best_iou"] for each in records) / len(records)

    path_counts: dict[str, int] = defaultdict(int)
    for each in records:
        path_counts[str(each.get("path_taken", "unknown"))] += 1

    return {
        "prompt_count": len(records),
        "avg_best_iou": avg_best_iou,
        "avg_mean_target_best_iou": avg_target,
        "avg_mean_predicted_best_iou": avg_predicted,
        "path_counts": dict(path_counts),
    }


def write_per_video_reports(per_video_records: dict[str, list[dict[str, Any]]], run_dir: Path) -> None:
    per_video_dir = run_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    for video_name, records in per_video_records.items():
        payload = {
            "video": video_name,
            "summary": summarize_results(records),
            "prompt_results": records,
        }
        report_path = per_video_dir / f"{Path(video_name).stem}.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_cumulative_summary(
    reports_dir: Path,
    all_existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    skipped_existing: int,
) -> None:
    payload = {
        "last_updated_utc": datetime.now(UTC).isoformat(),
        "cumulative": summarize_results(all_existing_records),
        "last_run": {
            "new_prompt_count": len(new_records),
            "skipped_existing_count": skipped_existing,
            "summary": summarize_results(new_records),
        },
    }

    summary_path = reports_dir / CUMULATIVE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_task(task: PromptTask, reports_dir: Path, generate_output_video: bool) -> dict[str, Any]:
    output_filename = f"eval_{task.video_path.stem}_{hashlib.md5(task.prompt.encode('utf-8')).hexdigest()[:10]}.mp4"
    output_path = str(reports_dir / "generated_clips" / output_filename)

    state = get_main_state(
        video_path=str(task.video_path),
        user_text=task.prompt,
        output_path=output_path,
        generate_output_video=generate_output_video,
    )

    final_state: Main_State = main_workflow.invoke(state)  # type: ignore

    predicted_timestamps = [
        (int(start), int(end)) for start, end in final_state.get("timestamps", [])
    ]

    iou_scores = compare_timestamps(predicted_timestamps, task.target_timestamps)
    route_details = dict(final_state.get("route_details", {}))
    route_details.pop("frames_scores", None)

    return {
        "sample_key": sample_key(task),
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "video": task.video_path.name,
        "video_path": task.video_path.as_posix(),
        "label_file": task.label_file.as_posix(),
        "prompt_id": task.prompt_id,
        "prompt": task.prompt,
        "path_taken": final_state.get("logical_path_choosen", "unknown"),
        "target_timestamps": [list(each) for each in task.target_timestamps],
        "predicted_timestamps": [list(each) for each in predicted_timestamps],
        "iou": iou_scores,
        "route_details": route_details,
        "matched_frames_count": len(final_state.get("matched_frames", [])),
    }


def main() -> None:
    args = parse_args()

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

    cumulative_jsonl = reports_dir / CUMULATIVE_RESULTS_FILENAME
    existing_keys = set() if args.force_recompute else load_existing_keys(cumulative_jsonl)

    pending_tasks: list[PromptTask] = []
    skipped_existing = 0
    for task in all_tasks:
        if sample_key(task) in existing_keys:
            skipped_existing += 1
            continue
        pending_tasks.append(task)

    if not pending_tasks:
        print("No new prompts to evaluate. All inputs are already present in cumulative report.")
        if cumulative_jsonl.exists():
            all_records = [
                json.loads(each)
                for each in cumulative_jsonl.read_text(encoding="utf-8").splitlines()
                if each.strip()
            ]
            upsert_cumulative_summary(reports_dir, all_records, [], skipped_existing)
        return

    pending_tasks.sort(key=lambda each: (each.video_path.as_posix(), each.prompt_id))

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = reports_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    per_video_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with tqdm(total=len(pending_tasks), desc="Evaluating prompts", unit="prompt") as progress_bar:
        for task in pending_tasks:
            record = evaluate_task(task, reports_dir, generate_output_video=args.generate_video)
            run_records.append(record)
            per_video_records[task.video_path.name].append(record)
            progress_bar.update(1)

    write_per_video_reports(per_video_records, run_dir)

    run_summary_payload = {
        "run_id": run_id,
        "started_utc": run_id,
        "finished_utc": datetime.now(UTC).isoformat(),
        "settings": {
            "videos_dir": videos_dir.as_posix(),
            "labels_dir": labels_dir.as_posix(),
            "generate_video": bool(args.generate_video),
            "force_recompute": bool(args.force_recompute),
            "max_prompts": args.max_prompts,
        },
        "counts": {
            "total_label_tasks": len(all_tasks),
            "evaluated_new": len(run_records),
            "skipped_existing": skipped_existing,
        },
        "summary": summarize_results(run_records),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(run_summary_payload, indent=2), encoding="utf-8"
    )

    with cumulative_jsonl.open("a", encoding="utf-8") as file_obj:
        for record in run_records:
            file_obj.write(json.dumps(record) + "\n")

    all_records = [
        json.loads(each)
        for each in cumulative_jsonl.read_text(encoding="utf-8").splitlines()
        if each.strip()
    ]
    upsert_cumulative_summary(reports_dir, all_records, run_records, skipped_existing)

    print(
        f"Evaluation complete. New records: {len(run_records)}, skipped existing: {skipped_existing}. "
        f"Run report: {run_dir.as_posix()}"
    )


if __name__ == "__main__":
    main()
