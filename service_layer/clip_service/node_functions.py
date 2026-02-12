from service_layer.clip_service.state import State
from langgraph.graph import END
import logging
from config import settings
from statistics import mean, stdev, median, quantiles

def generate_text_embeddings(state: State):
    
    text_embeddings = state["clip_processor"].encode_text_list(state["texts"])
    
    return { "text_embeddings": text_embeddings }


def generate_video_embeddings(state: State):
    
    video_embeddings_and_range_factory = lambda: state["clip_processor"].encode_frames(state["batch_frames_factory"]())
   
    return { "video_embeddings_and_range_factory": video_embeddings_and_range_factory }


def filter_only_matched_frames(state: State):
    
    matched_frames, frames_scores = state["clip_processor"].get_only_matched_frames(state["video_embeddings_and_range_factory"](), state["text_embeddings"])
    
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }


def reassess_scores(state: State):
    
    score_stats = state["score_stats"]
    
    if "mean" not in score_stats:
        
        # Comment these later
        score_stats["mean"] = mean((score for _, score in state["frames_scores"]))
        score_stats["median"] = median((score for _, score in state["frames_scores"]))
        score_stats["max"] = max(((score for _, score in state["frames_scores"])))
        score_stats["min"] = min(((score for _, score in state["frames_scores"])))
        
        
        score_stats["deciles"] = quantiles((score for _, score in state["frames_scores"]), n=10)
        
    
    logging.info("\n\n")
    logging.info("=" * 60)
    logging.info("Reassessing the matched frames....\n\n")
    logging.info(score_stats)
    logging.info("=" * 60)
    
    if score_stats["max"] < settings.CLIP_REASSESS_REJECT_THRESHOLD:
        
        logging.info("All scores are below the re-access reject threshold, the scene is probably not present")
        
        return { "reassessment_done": True, "score_stats": score_stats }
    
    
    matched_frames: list[tuple[int, int]] = [ frame_range for frame_range, score in state["frames_scores"] if score >= score_stats["deciles"][settings.CLIP_REASSESSMENT_DECILE_NUMBER - 1] ]
    
    return { "matched_frames": matched_frames, "reassessment_done": True, "score_stats": score_stats }


    
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