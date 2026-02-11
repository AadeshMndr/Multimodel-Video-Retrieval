from service_layer.yolo_service.state import State

def filter_only_matched_frames_that_match_any(state: State):
    
    matched_frames = state["yolo_processor"].get_only_matched_frames_that_match_any(state["batch_frames_factory"](), state["object_details"])
    
    return { "matched_frames": matched_frames }

def filter_only_matched_frames_that_match_all(state: State):
    
    matched_frames = state["yolo_processor"].get_only_matched_frames_that_match_all(state["batch_frames_factory"](), state["object_details"])
    
    return { "matched_frames": matched_frames }


def filter_only_matched_frames_in_canonical_form_POS(state: State):
    
    detection_objects_generator = state["yolo_processor"].get_detections_in_frames(state["batch_frames_factory"](), state["object_details"])
    
    groups = [ { object_name: state["object_details"][object_name] for object_name in each_group } for each_group in state["object_groups"] ]

    matched_frames = []
    
    for detection_object, start_last in detection_objects_generator:
        
        result = [state["yolo_processor"].match_if_any((detection_object, start_last), object_detail_dict) for object_detail_dict in groups]
        
        if all(result):
            matched_frames.append(start_last)
            
    return { "matched_frames": matched_frames }


def filter_only_matched_frames_in_canonical_form_SOP(state: State):
    
    detection_objects_generator = state["yolo_processor"].get_detections_in_frames(state["batch_frames_factory"](), state["object_details"])
    
    groups = [ { object_name: state["object_details"][object_name] for object_name in each_group } for each_group in state["object_groups"] ]

    matched_frames = []
    
    for detection_object, start_last in detection_objects_generator:
        
        result = [state["yolo_processor"].match_if_all((detection_object, start_last), object_detail_dict) for object_detail_dict in groups]
        
        if any(result):
            matched_frames.append(start_last)
            
    return { "matched_frames": matched_frames }
        
    

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

        
