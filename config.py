from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):

    DEVICE: str = "cpu"
    
    ENABLE_REASSESSMENT: bool = True

    PORT: int = 5050

    ############### Embedding store config ###############
   
   
    ENABLE_EMBEDDING_STORAGE: bool = True
    
    EMBEDDING_STORE_FILEPATH: str = "embeddings"
    CHUNKING_SIZE: int = 256

    CALCULATE_EMBEDDINGS_ON_PROCESSING: bool = False
    CALCULATE_EMBEDDINGS_ON_UPLOAD: bool = False

    
    ############### Video config ######################


    VIDEO_SAMPLING_RATE: int = 15
    
    # Lower number means more strict. 
    # PHASH_SIMILARITY_THRESHOLD: int = 12
    PHASH_SIMILARITY_THRESHOLD: int = 6
    
    # How many neighbouring frames to consider when merging and making a video.
    FRAME_NEIGHBOUR_RANGE_BEFORE: int = 30
    # FRAME_NEIGHBOUR_RANGE_BEFORE: int = 15
    FRAME_NEIGHBOUR_RANGE_AFTER: int = 60
    # FRAME_NEIGHBOUR_RANGE_AFTER: int = 15
    
    
    
    ################## CLIP config ####################
    
    CLIP_THRESHOLD: float = 0.230
    # CLIP_THRESHOLD: float = 0.192
    
    
    CLIP_REASSESSMENT_THRESHOLDS: list[float] = [0.210, 0.192]

    # above 90% (not including 90% though) (possible values: 1 to 9)
    CLIP_REASSESSMENT_DECILE_NUMBER: int | None = 9
    
    CLIP_REASSESS_REJECT_THRESHOLD: float = 0.05
    
    
    
    
    FRAME_BATCH_SIZE: int = 32
    
     
    MODEL_NAME: str = "ViT-L/14" # has embedding dimension -> 768
    # MODEL_NAME: str = "ViT-B/32" # has embedding dimension -> 512
    
    MAX_NUMBER_OF_MODIFIED_PROMPTS: int =  5  
    
    # This variable is not used by clip itself, but needs to be specified for Embedding to get stored.
    EMBEDDING_DIMENSION: int = 768 # Make sure this is correct as per the model choosen
    
    
    ################# YOLO config ######################
    
    # This threshold is used for "conf" in the model itself for detecting items
    YOLO_MIN_THRESHOLD: float = 0.100
    
    # This threshold is used for filtering the detected items in the first pass
    YOLO_MAX_USAGE_THRESHOLD: float = 0.350
    
    
    
    YOLO_REASSESSMENT_THRESHOLDS: list[float] = [0.250, 0.190]
    
    YOLO_REASSESSMENT_DECILE_NUMBER: int | None = 4
    
    YOLO_REASSESS_REJECT_THRESHOLD: float = 0.01
    
    
    YOLO_MODEL_NAME: str = "yoloe-26x-seg.pt"
    # YOLO_MODEL_NAME: str = "yoloe-11s-seg.pt"
    YOLO_BATCH_SIZE: int = 32
    
    # YOLO_MAX_USAGE_THRESHOLD: float = 0.950

    
    MAX_NUM_OF_SYNONYMS: int = 5

    
    
    #################### LLM config ####################
    
    
    LLM_MODEL_NAME: str = "groq/compound"
    GROQ_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
    
    
settings = Settings()