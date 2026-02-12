from service_layer.yolo_service.state import State
from types_and_schemas.yolo_detection_types import ScoreData

def filter_only_matched_frames_that_match_any(state: State):
    
    matched_frames, frames_scores = state["yolo_processor"].get_only_matched_frames_that_match_any(state["batch_frames_factory"](), state["object_details"])
    
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }

def filter_only_matched_frames_that_match_all(state: State):
    
    matched_frames, frames_scores = state["yolo_processor"].get_only_matched_frames_that_match_all(state["batch_frames_factory"](), state["object_details"])
    
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }


def filter_only_matched_frames_in_canonical_form_POS(state: State):
    
    detection_objects_generator = state["yolo_processor"].get_detections_in_frames(state["batch_frames_factory"](), state["object_details"])
    
    groups = [ { object_name: state["object_details"][object_name] for object_name in each_group } for each_group in state["object_groups"] ]

    matched_frames = []
    frames_scores: list[ScoreData] = []
    
    for (detection_object, start_last), frame_score in detection_objects_generator:
        
        result = [state["yolo_processor"].match_if_any((detection_object, start_last), object_detail_dict) for object_detail_dict in groups]
        
        frames_scores.append(frame_score)
        
        if all(result):
            matched_frames.append(start_last)
            
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }


def filter_only_matched_frames_in_canonical_form_SOP(state: State):
    
    detection_objects_generator = state["yolo_processor"].get_detections_in_frames(state["batch_frames_factory"](), state["object_details"])
    
    groups = [ { object_name: state["object_details"][object_name] for object_name in each_group } for each_group in state["object_groups"] ]

    matched_frames = []
    
    frames_scores: list[ScoreData] = []
    
    for (detection_object, start_last), frame_score in detection_objects_generator:
        
        result = [state["yolo_processor"].match_if_all((detection_object, start_last), object_detail_dict) for object_detail_dict in groups]
        
        frames_scores.append(frame_score)
        
        if any(result):
            matched_frames.append(start_last)
            
            
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }
        
    

def decision_node(state: State):
    
    if len(state["object_groups"]) == 1:

        if state["canonical_form"] == "POS":
            return "match_any"
        else:
            return "match_all"
        
    else:
        
        if state["canonical_form"] == "POS":
            return "match_POS"
        else:
            return "match_SOP"

        
