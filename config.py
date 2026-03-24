from pydantic_settings import BaseSettings
from typing import Literal
import torch

def get_optimal_device() -> str:
    """Detect the best available device for GPU acceleration"""
    if torch.cuda.is_available():
        return "cuda"  # For NVIDIA GPUs
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"   # For Apple Silicon Mac GPUs
    else:
        return "cpu"

class Settings(BaseSettings):

    DEVICE: str = get_optimal_device()
    
    ENABLE_REASSESSMENT: bool = True

    PORT: int = 5050

    ############### Embedding store config ###############
   
   
    ENABLE_EMBEDDING_STORAGE: bool = True
    
    EMBEDDING_STORE_FILEPATH: str = "embeddings"
    CHUNKING_SIZE: int = 256

    # CALCULATE_EMBEDDINGS_ON_PROCESSING: bool = False
    CALCULATE_EMBEDDINGS_ON_PROCESSING: bool = True
    # CALCULATE_EMBEDDINGS_ON_UPLOAD: bool = True
    CALCULATE_EMBEDDINGS_ON_UPLOAD: bool = False 

    
    ############### Video config ######################


    VIDEO_SAMPLING_RATE: int = 15
    
    # Lower number means more strict. 
    # PHASH_SIMILARITY_THRESHOLD: int = 12
    PHASH_SIMILARITY_THRESHOLD: int = 6
    
    # How many neighbouring frames to consider when merging and making a video.
    FRAME_NEIGHBOUR_RANGE_BEFORE: int = 30
    # FRAME_NEIGHBOUR_RANGE_BEFORE: int = 15
    FRAME_NEIGHBOUR_RANGE_AFTER: int = 30
    # FRAME_NEIGHBOUR_RANGE_AFTER: int = 15
    
    
    
    ################## CLIP config ####################
    

    CLIP_THRESHOLD: float = 0.250
    # CLIP_THRESHOLD: float = 0.230
    # CLIP_THRESHOLD: float = 0.192
    
    
    CLIP_REASSESSMENT_THRESHOLDS: list[float] = [0.230, 210, 0.192]

    # above 90% (not including 90% though) (possible values: 0 to 8) -> 8 means: above 90%
    CLIP_REASSESSMENT_DECILE_NUMBER: int | None = 8
    
    CLIP_REASSESS_REJECT_THRESHOLD: float = 0.05
    
    
    
    
    FRAME_BATCH_SIZE: int = 64 if get_optimal_device() != "cpu" else 32
    
     
    MODEL_NAME: str = "ViT-L/14" # has embedding dimension -> 768
    # MODEL_NAME: str = "ViT-B/32" # has embedding dimension -> 512
    
    MAX_NUMBER_OF_MODIFIED_PROMPTS: int =  5  
    
    # This variable is not used by clip itself, but needs to be specified for Embedding to get stored.
    EMBEDDING_DIMENSION: int = 768 # Make sure this is correct as per the model choosen
    
    
    ################# YOLO config ######################
    
    # This threshold is used for "conf" in the model itself for detecting items
    YOLO_MIN_THRESHOLD: float = 0.100
    
    # This threshold is used for filtering the detected items in the first pass
    YOLO_MAX_USAGE_THRESHOLD: float = 0.700
    
    
    
    YOLO_REASSESSMENT_THRESHOLDS: list[float] = [0.500, 0.350, 0.250, 0.190]
    
    YOLO_REASSESSMENT_DECILE_NUMBER: int | None = 4
    
    YOLO_REASSESS_REJECT_THRESHOLD: float = 0.01
    
    
    YOLO_MODEL_NAME: str = "yoloe-26x-seg.pt"
    # YOLO_MODEL_NAME: str = "yoloe-11s-seg.pt"
    YOLO_BATCH_SIZE: int = 64 if get_optimal_device() != "cpu" else 32
    
    # YOLO_MAX_USAGE_THRESHOLD: float = 0.950

    
    MAX_NUM_OF_SYNONYMS: int = 5

    
    
    #################### LLM config ####################
    
    
    LLM_MODEL_NAME: str = "groq/compound"
    GROQ_API_KEY: str = ""

    #################### Audio Retrieval ####################

    AUDIO_INDEX_DIR: str = "audio_indexes"
    AUDIO_WHISPER_MODEL_SIZE: str = "small"
    AUDIO_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    AUDIO_CHUNK_WINDOW_SIZE: int = 20
    AUDIO_CHUNK_OVERLAP_SIZE: int = 5
    AUDIO_TOP_K: int = 10
    AUDIO_MERGE_GAP: float = 8.0
    AUDIO_FORCE_REINDEX: bool = False
    CALCULATE_AUDIO_INDEX_ON_UPLOAD: bool = False
    
    class Config:
        env_file = ".env"
    
    
settings = Settings()
