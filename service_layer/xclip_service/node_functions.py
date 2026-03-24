import logging

from langgraph.graph import END

from config import settings
from service_layer.xclip_service.state import State


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _filter_and_merge_by_threshold(state: State, threshold: float) -> list[tuple[int, int]]:
    matched_ranges = [
        frame_range
        for frame_range, score in zip(state["frame_ranges"], state["window_scores"])
        if score >= threshold
    ]
    return state["xclip_processor"]._merge_ranges(matched_ranges, fps=state["fps"])


def find_temporal_matches(state: State):
    frame_ranges, window_scores, score_stats = state["xclip_processor"].compute_window_scores(
        sampled_frames_factory=state["sampled_frames_factory"],
        texts=state["texts"],
        fps=state["fps"],
    )

    matched_frames = [
        frame_range
        for frame_range, score in zip(frame_ranges, window_scores)
        if score >= settings.XCLIP_THRESHOLD
    ]
    merged_matches = state["xclip_processor"]._merge_ranges(matched_frames, fps=state["fps"])

    score_stats["used_threshold"] = settings.XCLIP_THRESHOLD

    logging.info("=" * 60)
    logging.info("XCLIP initial matching complete")
    logging.info("Initial threshold: %.3f", settings.XCLIP_THRESHOLD)
    logging.info("Window count: %s", len(frame_ranges))
    logging.info("Matched windows: %s", score_stats.get("matched_window_count", 0))
    logging.info("Matched merged ranges: %s", len(merged_matches))
    logging.info("Score stats: mean=%s median=%s max=%s min=%s", score_stats.get("mean"), score_stats.get("median"), score_stats.get("max"), score_stats.get("min"))
    logging.info("=" * 60)

    return {
        "frame_ranges": frame_ranges,
        "window_scores": window_scores,
        "matched_frames": merged_matches,
        "score_stats": score_stats,
        "reassessment_count": 0,
    }


def reassess_matches(state: State):
    threshold = settings.XCLIP_REASSESSMENT_THRESHOLDS[state["reassessment_count"]]
    matched_window_count = sum(
        1
        for score in state["window_scores"]
        if score >= threshold
    )
    matched_frames = _filter_and_merge_by_threshold(state, threshold)

    score_stats = dict(state["score_stats"])
    score_stats["used_threshold"] = threshold
    score_stats["matched_window_count"] = matched_window_count

    logging.info("Reassessing XCLIP with threshold: %.3f", threshold)
    logging.info("Reassessment pass: %s/%s", state["reassessment_count"] + 1, len(settings.XCLIP_REASSESSMENT_THRESHOLDS))
    logging.info("Matched windows after reassessment: %s", matched_window_count)
    logging.info("Matched merged ranges after reassessment: %s", len(matched_frames))

    return {
        "matched_frames": matched_frames,
        "reassessment_count": state["reassessment_count"] + 1,
        "score_stats": score_stats,
    }


def is_reassessment_required(state: State):
    reassessment_possible = (
        settings.ENABLE_REASSESSMENT
        and state["reassessment_count"] < len(settings.XCLIP_REASSESSMENT_THRESHOLDS)
    )

    if not reassessment_possible:
        logging.info("XCLIP reassessment complete or disabled. Proceeding without further reassessment.")
        return END

    if len(state["matched_frames"]) == 0 and len(state["window_scores"]) > 0:
        logging.info("No XCLIP matches found at threshold %.3f, triggering reassessment.", state["score_stats"].get("used_threshold", settings.XCLIP_THRESHOLD))
        return "re-assess"

    if len(state["matched_frames"]) > 0:
        logging.info("XCLIP produced matches at threshold %.3f, no reassessment needed.", state["score_stats"].get("used_threshold", settings.XCLIP_THRESHOLD))

    return END
