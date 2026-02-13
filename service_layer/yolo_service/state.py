from typing import TypedDict, Callable, Literal
from infrastructure.yolo_processor import YOLO_Processor
from types_and_schemas.video_types import Generator_Batch_Image_Range
from types_and_schemas.yolo_detection_types import ScoreData, Generator_Range_Detection_Count_Score


class State(TypedDict):
    
    object_details: dict[str, tuple[float, float]]
    
    batch_frames_factory: Callable[[], Generator_Batch_Image_Range]

    object_groups: list[list[str]]
    
    canonical_form: Literal["SOP", "POS"]
    
    yolo_processor: YOLO_Processor

    frames_scores: list[ScoreData]
    
    reassessment_count: int
    
    reassess_detection_object_generator_factory: Callable[[], Generator_Range_Detection_Count_Score] | None

    score_stats: dict
    
    matched_frames: list[tuple[int, int]]


def get_state(
    yolo_processor: YOLO_Processor,
    object_details: dict[str, tuple[float, float]], 
    batch_frames_factory: Callable[[], Generator_Batch_Image_Range],  
    object_groups: list[list[str]], 
    canonical_form: Literal["SOP", "POS"]):
    
    return State( 
        yolo_processor=yolo_processor,
        object_details=object_details,
        batch_frames_factory=batch_frames_factory,
        object_groups=object_groups,
        canonical_form=canonical_form,
        matched_frames=[],
        frames_scores=[],
        score_stats={},
        reassessment_count=0,
        reassess_detection_object_generator_factory=None
    )
    
    