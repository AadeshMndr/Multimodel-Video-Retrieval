from ultralytics import YOLO # type: ignore
from config import settings
import logging
from types_and_schemas.video_types import Generator_Batch_Image_Range

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class YOLO_Processor:
    
    def __init__(self, model_name: str = settings.YOLO_MODEL_NAME):
        
        self.model = YOLO(model=model_name)
    
    
    def get_only_matched_frames(self, batch_frame_range: Generator_Batch_Image_Range, object_names: list[str]) -> list[tuple[int, int]]:
        
        logging.info("Processing begins...")
        
        self.model.set_classes(object_names) # type: ignore
        
        matched_frames_data: list[tuple[int, int]] = []
        
        # batch_number = 1
        
        for batch_frames, start_last_data in batch_frame_range:
            
            results = self.model(
                source=batch_frames,
                batch=len(batch_frames)
            )
            
            # Complete the logic from here
            confidence_scores = results.boxes.conf 
            
            
            
            # logging.info("\n")
            # logging.info(f"Processing batch {batch_number}")
            # batch_number += 1
            
            # matched_data = self.match_frames_and_text(batch_frame_tensor, text_embeddings, start_last_data)
            matched_data = []
            
            # logging.info(f"Found {len(matched_data)} matching frame ranges")
            # logging.info("\n")
            matched_frames_data.extend(matched_data)
            
        logging.info(f"Number of Matched Frame Groups: {len(matched_frames_data)}")
            
        return matched_frames_data