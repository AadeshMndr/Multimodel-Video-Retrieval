from service_layer.clip_service.state import State
from langgraph.graph import END
import logging
from config import settings
from statistics import mean, median, quantiles
import numpy as np

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
        for key in score_stats.keys():
            logging.info(f"{key}: {score_stats[key]}")
        logging.info("=" * 60)
    
    
    if score_stats["max"] < settings.CLIP_REASSESS_REJECT_THRESHOLD:
        
        logging.info("All scores are below the re-access reject threshold, the scene is probably not present")
        
        return { "reassessment_count": len(settings.CLIP_REASSESSMENT_THRESHOLDS) + 1, "score_stats": score_stats }
    
    
    if state["reassessment_count"] < len(settings.CLIP_REASSESSMENT_THRESHOLDS):
        
        threshold = settings.CLIP_REASSESSMENT_THRESHOLDS[state["reassessment_count"]]
        
        # Again, here I'm intentionally ignoring that threshold could be higher than set quantile,
        # because, we might need that whole range of frames as well instead of the best top 90% (or whatever value is set)

        logging.info(f"Reassessing with threshold: {threshold}")
        
        matched_frames: list[tuple[int, int]] = [ frame_range for frame_range, score in state["frames_scores"] if score >= threshold ]
        
        return { "reassessment_count": state["reassessment_count"] + 1, "matched_frames": matched_frames, "score_stats": score_stats }
    
    
    if settings.CLIP_REASSESSMENT_DECILE_NUMBER is None:
        
        return { "reassessment_count": len(settings.CLIP_REASSESSMENT_THRESHOLDS) + 1, "score_stats": score_stats }
    
    
    logging.info(f"Reassessing with threshold (decile): {score_stats["deciles"][settings.CLIP_REASSESSMENT_DECILE_NUMBER or 0 - 1]}")
    
    matched_frames: list[tuple[int, int]] = [ frame_range for frame_range, score in state["frames_scores"] if score >= score_stats["deciles"][settings.CLIP_REASSESSMENT_DECILE_NUMBER or 0 - 1] ]
    
    return { "matched_frames": matched_frames, "reassessment_count": len(settings.CLIP_REASSESSMENT_THRESHOLDS) + 1, "score_stats": score_stats }


    
def is_reassessment_required(state: State):
    
    reassessment_possible = state["reassessment_count"] < (len(settings.CLIP_REASSESSMENT_THRESHOLDS) + 1) and settings.ENABLE_REASSESSMENT
    
    if not reassessment_possible:
        return END
    
    
    if len(state["matched_frames"]) == 0:
        
        if len(state["frames_scores"]) > 0:
            
            return "re-assess"
        
        else:
            
            logging.error("There was no values in 'frames_scores', the scores were not being stored, so skipping reassessment")
            
    return END
        
def store_video_embeddings(state: State):
    
    for batch_frame_embedding, batch_start_last_data in state["video_embeddings_and_range_factory"]():
        
        batch_frame_embedding_numpy = batch_frame_embedding.to("cpu").numpy()
        
        batch_start_last_numpy = np.array(batch_start_last_data)
        
        state["clip_processor"].embedding_store.store_batch_embeddings(batch_frame_embedding_numpy, batch_start_last_numpy)
        
    
    