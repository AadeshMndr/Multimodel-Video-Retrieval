from typing import Callable, TypedDict

from infrastructure.xclip_processor import XCLIP_Processor
from types_and_schemas.video_types import Generator_Image_Range


class State(TypedDict):
    texts: list[str]
    sampled_frames_factory: Callable[[], Generator_Image_Range]
    fps: float
    xclip_processor: XCLIP_Processor
    matched_frames: list[tuple[int, int]]
    frame_ranges: list[tuple[int, int]]
    window_scores: list[float]
    reassessment_count: int
    score_stats: dict


def get_state(
    texts: list[str],
    sampled_frames_factory: Callable[[], Generator_Image_Range],
    fps: float,
    xclip_processor: XCLIP_Processor,
) -> State:
    return State(  # type: ignore
        texts=texts,
        sampled_frames_factory=sampled_frames_factory,
        fps=fps,
        xclip_processor=xclip_processor,
        matched_frames=[],
        frame_ranges=[],
        window_scores=[],
        reassessment_count=0,
        score_stats={},
    )
