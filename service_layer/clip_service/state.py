from typing import TypedDict, Callable
from torch import Tensor
from infrastructure.clip_processor import CLIP_Processor
from types_and_schemas.video_types import Generator_Batch_Image_Range, Generator_Batch_Tensor_Range

class State(TypedDict):
    texts: list[str]
    
    batch_frames_factory: Callable[[], Generator_Batch_Image_Range]
    
    clip_processor: CLIP_Processor
    
    text_embeddings: Tensor
    
    video_embeddings_and_range_factory: Callable[[], Generator_Batch_Tensor_Range]

    frames_scores: list[float]
    
    matched_frames: list[tuple[int, int]]
    
def get_state(texts: list[str], clip_processor: CLIP_Processor, batch_frames_factory: Callable[[], Generator_Batch_Image_Range]):
    
    return State(                      # type: ignore
        texts=texts,
        clip_processor=clip_processor,
        batch_frames_factory=batch_frames_factory
    )
   
    