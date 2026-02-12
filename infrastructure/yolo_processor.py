from ultralytics import YOLO # type: ignore
from config import settings
import logging
from types_and_schemas.video_types import Generator_Batch_Image_Range
from types_and_schemas.yolo_detection_types import Generator_Range_Detection_Count, Detection_Range_Count, ScoreData


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class YOLO_Processor:
    
    def __init__(self, model_name: str = settings.YOLO_MODEL_NAME, device: str = settings.DEVICE):
        
        self.model = YOLO(model=model_name)
        self.device = device
    
   
    def get_detections_in_frames(self, batch_frame_range: Generator_Batch_Image_Range, object_details: dict[str, tuple[float, float]]) -> Generator_Range_Detection_Count:
       
        logging.info("Processing begins (Detection) ...")
        
        self.model.set_classes(list(object_details.keys())) # type: ignore
        
        # batch_number = 1
        
        for batch_frames, start_last_data in batch_frame_range:
            
            results = self.model(
                source=batch_frames,
                batch=len(batch_frames),
                conf=settings.YOLO_MIN_THRESHOLD,
                device=self.device,
                verbose=False
            )
            
            
            for result, each_start_last_data in zip(results, start_last_data):
                
                object_detections = {}
                # object_scores: dict[str, list[float]] = {}
                
                for cls in result.boxes.cls:
                    
                    cls_index = cls.item()
                    
                    class_name = result.names[cls_index] # type: ignore
                    
                    object_detections[class_name] = object_detections.get(class_name, 0) + 1
                    
            
                yield (object_detections, each_start_last_data)
                
                    
    def match_if_any(self, detection_object: Detection_Range_Count,  object_details: dict[str, tuple[float, float]]) -> bool:
        
        object_detections, _ = detection_object
        
        for object_name, ( low_count, high_count ) in object_details.items():
                    
            # print(object_detections, object_name, low_count, high_count)
            if low_count <= object_detections.get(object_name, 0) <= high_count:
                return True
      
                
        return False
    
    
    def match_if_all(self, detection_object: Detection_Range_Count,  object_details: dict[str, tuple[float, float]]) -> bool:
        
        object_detections, _ = detection_object        

        for object_name, ( low_count, high_count ) in object_details.items():
                    
            if high_count < object_detections.get(object_name, 0) or object_detections.get(object_name, 0) < low_count:
                return False
                
        return True

        
   
    
    def get_only_matched_frames_that_match_any(self, batch_frame_range: Generator_Batch_Image_Range, object_details: dict[str, tuple[float, float]]) -> tuple[list[tuple[int, int]], list[ScoreData]]:
        
        logging.info("Processing begins...")
        
        self.model.set_classes(list(object_details.keys())) # type: ignore
        
        matched_frames_data: list[tuple[int, int]] = []

        frames_scores: list[ScoreData] = []
        
        # batch_number = 1
        
        for batch_frames, start_last_data in batch_frame_range:
            
            results = self.model(
                source=batch_frames,
                batch=len(batch_frames),
                conf=settings.YOLO_MIN_THRESHOLD,
                device=self.device,
                verbose=False
            )
            
            
            for result, each_frame_group_start_last_data in zip(results, start_last_data):
                
                object_detections = {}
                
                object_scores: dict[str, list[float]] = {}
                
                for conf, cls in zip(result.boxes.conf, result.boxes.cls):
                    
                    cls_index = cls.item()
                    
                    conf_score = conf.item()
                    
                    class_name = result.names[cls_index] # type: ignore
                    
                    if class_name not in object_scores:
                        object_scores[class_name] = [ conf_score ]
                    else:
                        object_scores[class_name].append(conf_score)
                    
                    
                    if conf_score >= settings.YOLO_MAX_USAGE_THRESHOLD:
                        object_detections[class_name] = object_detections.get(class_name, 0) + 1
                        
                frames_scores.append((each_frame_group_start_last_data, object_scores))
                
            
                for object_name, ( low_count, high_count ) in object_details.items():
                    
                    if low_count <= object_detections.get(object_name, 0) <= high_count:
                        # print(each_frame_group_start_last_data, object_detections, object_name, low_count, high_count)
                        matched_frames_data.append(each_frame_group_start_last_data)
                        break
                    
                
            
        logging.info(f"Number of Matched Frame Groups: {len(matched_frames_data)}")
            
        return matched_frames_data, frames_scores
    
    
    def get_only_matched_frames_that_match_all(self, batch_frame_range: Generator_Batch_Image_Range, object_details: dict[str, tuple[float, float]]) -> tuple[list[tuple[int, int]], list[ScoreData]]:
        
        logging.info("Processing begins...")
        
        self.model.set_classes(list(object_details.keys())) # type: ignore
        
        matched_frames_data: list[tuple[int, int]] = []
        
        frames_scores: list[ScoreData] = []
        
        # batch_number = 1
        
        for batch_frames, start_last_data in batch_frame_range:
            
            results = self.model(
                source=batch_frames,
                batch=len(batch_frames),
                conf=settings.YOLO_MIN_THRESHOLD,
                device=self.device,
                verbose=False
            )
            
            
            for result, each_frame_group_start_last_data in zip(results, start_last_data):
                
                object_detections = {}
                
                object_scores = {}
                
                for conf, cls in zip(result.boxes.conf, result.boxes.cls):
                    
                    cls_index = cls.item()
                    
                    conf_score = conf.item()
                    
                    class_name = result.names[cls_index] # type: ignore
                    
                    if class_name not in object_scores:
                        object_scores[class_name] = [ conf_score ]
                    else:
                        object_scores[class_name].append(conf_score)
                    
                    
                    if conf_score >= settings.YOLO_MAX_USAGE_THRESHOLD:
                        object_detections[class_name] = object_detections.get(class_name, 0) + 1
                    
                frames_scores.append((each_frame_group_start_last_data, object_scores))
            
                for object_name, ( low_count, high_count ) in object_details.items():
                    
                    # print(each_frame_group_start_last_data, object_detections, object_name, low_count, high_count)
                    if high_count < object_detections.get(object_name, 0) or object_detections.get(object_name, 0) < low_count:
                        break
                
                else:    
                    matched_frames_data.append(each_frame_group_start_last_data)
                
            
        logging.info(f"Number of Matched Frame Groups: {len(matched_frames_data)}")
            
        return matched_frames_data, frames_scores