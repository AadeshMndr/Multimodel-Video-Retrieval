from typing import TypedDict
from service_layer.video_service.state import State as VideoState
from service_layer.llm_service.state import All_Path
from threading import Event
from infrastructure.h5py_storage import Embedding_Store
from config import settings

class Main_State(TypedDict):
    video_path: str 
    user_text: str
    logical_path_choosen: All_Path
    output_path: str
    video_state: VideoState
    matched_frames: list[tuple[int, int]]
    timestamps: list[tuple[int, int]]
    embedding_store: Embedding_Store

    
    video_creation_event: Event | None
    
def get_main_state(video_path: str, user_text: str, output_path: str, embedding_store_name: str = settings.EMBEDDING_STORE_FILEPATH, chunking_size: int = settings.CHUNKING_SIZE, embedding_dimension: int = settings.EMBEDDING_DIMENSION):
    embedding_store = Embedding_Store(
        file_name=embedding_store_name,
        chunking_size=chunking_size,
        dataset_name=video_path,
        embedding_dimension=embedding_dimension
    )
    
    return Main_State( # type: ignore
        video_path=video_path,
        user_text=user_text,
        output_path=output_path,
        embedding_store=embedding_store,
        video_creation_event=None
    )