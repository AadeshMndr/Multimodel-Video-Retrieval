from service_layer.clip_service.state import State

def generate_text_embeddings(state: State):
    
    text_embeddings = state["clip_processor"].encode_text_list(state["texts"])
    
    return { "text_embeddings": text_embeddings }


def generate_video_embeddings(state: State):
    
    video_embeddings_and_range_factory = lambda: state["clip_processor"].encode_frames(state["batch_frames_factory"]())
   
    return { "video_embeddings_and_range_factory": video_embeddings_and_range_factory }


def filter_only_matched_frames(state: State):
    
    matched_frames, frames_scores = state["clip_processor"].get_only_matched_frames(state["video_embeddings_and_range_factory"](), state["text_embeddings"])
    
    return { "matched_frames": matched_frames, "frames_scores": frames_scores }