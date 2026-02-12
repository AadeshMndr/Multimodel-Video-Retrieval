from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):

    DEVICE: str = "cpu"
    
    ENABLE_REASSESSMENT: bool = True


    
    ############### Video config ######################


    VIDEO_SAMPLING_RATE: int = 15
    
    # Lower number means more strict. 
    PHASH_SIMILARITY_THRESHOLD: int = 6
    
    # How many neighbouring frames to consider when merging and making a video.
    FRAME_NEIGHBOUR_RANGE_BEFORE: int = 30
    # FRAME_NEIGHBOUR_RANGE_BEFORE: int = 15
    FRAME_NEIGHBOUR_RANGE_AFTER: int = 60
    # FRAME_NEIGHBOUR_RANGE_AFTER: int = 15
    
    
    
    ################## CLIP config ####################
    
    
    # above 70% (not including 70% though) (possible values: 1 to 9)
    CLIP_REASSESSMENT_DECILE_NUMBER: int = 7
    
    FRAME_BATCH_SIZE: int = 32
    
    
    
    MODEL_NAME: str = "ViT-L/14" 
    CLIP_THRESHOLD: float = 0.230
    # CLIP_THRESHOLD: float = 0.192
    MAX_NUMBER_OF_MODIFIED_PROMPTS: int =  5  
    
    CLIP_REASSESS_REJECT_THRESHOLD: float = 0.05
    
    
    ################# YOLO config ######################
    
    YOLO_REASSESSMENT_DECILE_NUMBER: int = 4
    
    YOLO_MODEL_NAME: str = "yoloe-26x-seg.pt"
    # YOLO_MODEL_NAME: str = "yoloe-11s-seg.pt"
    YOLO_BATCH_SIZE: int = 32
    
    YOLO_MIN_THRESHOLD: float = 0.100
    YOLO_MAX_USAGE_THRESHOLD: float = 0.350
    # YOLO_MAX_USAGE_THRESHOLD: float = 0.950

    
    MAX_NUM_OF_SYNONYMS: int = 5

    YOLO_REASSESS_REJECT_THRESHOLD: float = 0.01
    
    
    #################### LLM config ####################
    
    
    LLM_MODEL_NAME: str = "groq/compound"
    GROQ_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
    
    
settings = Settings()