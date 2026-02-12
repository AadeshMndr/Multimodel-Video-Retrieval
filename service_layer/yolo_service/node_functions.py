from service_layer.yolo_service.state import State
from types_and_schemas.yolo_detection_types import ScoreData, Generator_Range_Detection_Count_Score
from config import settings
from statistics import mean, median, quantiles, stdev
import logging
from langgraph.graph import END
from typing import Generator

def filter_only_matched_frames_that_match_any(state: State):
    
    matched_frames, frames_scores = state["yolo_processor"].get_only_matched_frames_that_match_any(state["batch_frames_factory"](), state["object_details"])
    
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }

def filter_only_matched_frames_that_match_all(state: State):
    
    matched_frames, frames_scores = state["yolo_processor"].get_only_matched_frames_that_match_all(state["batch_frames_factory"](), state["object_details"])
    
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }


def filter_only_matched_frames_in_canonical_form_POS(state: State):
    
    detection_objects_generator = state["yolo_processor"].get_detections_in_frames(state["batch_frames_factory"](), state["object_details"])
    
    if state["reassess_detection_object_generator_factory"] is not None:
        logging.info(f"Using the generator given after re-assessment...")
        detection_objects_generator = state["reassess_detection_object_generator_factory"]()
    
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
    
    if state["reassess_detection_object_generator_factory"] is not None:
        logging.info(f"Using the generator given after re-assessment...")
        detection_objects_generator = state["reassess_detection_object_generator_factory"]()
    
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
    
    if state["reassessment_done"] and state["reassess_detection_object_generator_factory"] is None:
        return END
        
    if state["canonical_form"] == "POS":
        return "match_POS"
    else:
        return "match_SOP"

        
def reassess_scores(state: State):
    
    score_stats = state["score_stats"]
    
    
    if "done" not in score_stats:
        
        score_dict_list: dict[str, list[float]] = {}    
        
        for class_name in state["object_details"]:
            
            score_dict_list[class_name] = []
            
            for _, score_dict in state["frames_scores"]:
                
                score_dict_list[class_name].extend(score_dict.get(class_name, []))
                
            if len(score_dict_list[class_name]) == 0:
                score_dict_list[class_name] = [0] * 10
                
        # Comment these later
        score_stats["mean"] = { class_name: mean(score_dict_list[class_name]) for class_name in state["object_details"] }
        score_stats["median"] = { class_name: median(score_dict_list[class_name]) for class_name in state["object_details"] }
        score_stats["max"] = { class_name: max(score_dict_list[class_name]) for class_name in state["object_details"] }
        score_stats["min"] = { class_name: min(score_dict_list[class_name]) for class_name in state["object_details"] }
    
    
        score_stats["decile"] = { class_name: quantiles(score_dict_list[class_name], n=10) for class_name in state["object_details"] }
        
        score_stats["done"] = True
        
        
    logging.info("\n\n")
    logging.info("=" * 60)
    logging.info("Reassessing the matched frames....\n\n")
    for key in score_stats.keys():
        logging.info(f"{key}: {score_stats[key]}")
    logging.info("=" * 60)
        
    if all([score_stats["max"][class_name] < settings.YOLO_REASSESS_REJECT_THRESHOLD for class_name in score_stats["max"].keys()]):
        
        logging.info("All scores are below the re-access reject threshold, the scene is probably not present")
        
        return { "reassessment_done": True }

    
    
    def detection_objects_generator_factory() -> Generator_Range_Detection_Count_Score:
    
        for start_last_range, score_dict in state["frames_scores"]:
            
            object_detection = { class_name: len([score for score in score_dict[class_name] if score > score_stats["decile"][class_name][settings.YOLO_REASSESSMENT_DECILE_NUMBER - 1]]) for class_name in score_dict.keys() }
            
            yield ((object_detection, start_last_range), (start_last_range, {}))

    
    
    return { "reassessment_done": True, "reassess_detection_object_generator_factory": detection_objects_generator_factory }


    
def is_reassessment_required(state: State):
    
    reassessment_possible = not state["reassessment_done"] and settings.ENABLE_REASSESSMENT
    
    if not reassessment_possible:
        return END
    
    
    if len(state["matched_frames"]) == 0:
        
        if len(state["frames_scores"]) > 0:
            
            return "re-assess"
        
        else:
            
            logging.error("There was no values in 'frames_scores', the scores were not being stored, so skipping reassessment")
            
    return END