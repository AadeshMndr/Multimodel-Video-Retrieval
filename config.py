from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    DEVICE: str = "cpu"


    VIDEO_SAMPLING_RATE: int = 15
    
    # Lower number means more strict. 
    PHASH_SIMILARITY_THRESHOLD: int = 6
    
    # How many neighbouring frames to consider when merging and making a video.
    FRAME_NEIGHBOUR_RANGE_BEFORE: int = 30
    # FRAME_NEIGHBOUR_RANGE_BEFORE: int = 15
    FRAME_NEIGHBOUR_RANGE_AFTER: int = 60
    # FRAME_NEIGHBOUR_RANGE_AFTER: int = 15
    FRAME_BATCH_SIZE: int = 32
    
    
    
    MODEL_NAME: str = "ViT-L/14" 
    CLIP_THRESHOLD: float = 0.260
    # CLIP_THRESHOLD: float = 0.192
    MAX_NUMBER_OF_MODIFIED_PROMPTS: int =  5  
    
    
    ################# YOLO config ######################
    
    YOLO_MODEL_NAME: str = "yoloe-26x-seg.pt"
    # YOLO_MODEL_NAME: str = "yoloe-11s-seg.pt"
    YOLO_BATCH_SIZE: int = 32
    
    YOLO_THRESHOLD: float = 0.250

    
    MAX_NUM_OF_SYNONYMS: int = 5
    LLM_MODEL_NAME: str = "groq/compound"
    GROQ_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
    
    
settings = Settings()