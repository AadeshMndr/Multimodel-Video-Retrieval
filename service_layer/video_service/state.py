from typing import TypedDict, Callable
from infrastructure.video_processor import Video_Processor
from types_and_schemas.video_types import Generator_Image_Range, Generator_Batch_Image_Range, Generator_Generic_Range

class State(TypedDict):
    
    output_path: str
    
    video_processor: Video_Processor 
   
    batch_size: int
    
    sampled_generator_factory: Callable[[], Generator_Image_Range]
    
    reduced_generator_factory: Callable[[], Generator_Image_Range]

    batched_generator_factory: Callable[[], Generator_Batch_Image_Range]
    
    matched_frame_range: list[tuple[int, int]]
    
    expanded_frame_range_generator_factory: Callable[[], Generator_Generic_Range]
    

def get_state(output_path: str, video_processor: Video_Processor, batch_size: int) -> State:
    
    return State(                              # type: ignore
        output_path=output_path,
        video_processor=video_processor,
        batch_size=batch_size
    )