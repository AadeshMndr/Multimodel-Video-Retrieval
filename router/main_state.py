from typing import TypedDict
from service_layer.video_service.state import State as VideoState
from service_layer.llm_service.state import All_Path
from threading import Event

class Main_State(TypedDict):
    video_path: str 
    user_text: str
    logical_path_choosen: All_Path
    output_path: str
    video_state: VideoState
    matched_frames: list[tuple[int, int]]
    timestamps: list[tuple[int, int]]
    
    video_creation_event: Event | None
    
def get_main_state(video_path: str, user_text: str, output_path: str):
    return Main_State( # type: ignore
        video_path=video_path,
        user_text=user_text,
        output_path=output_path
    )