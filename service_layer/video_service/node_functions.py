from service_layer.video_service.state import State

def sample_frames(state: State):
    
    sampled_generator_factory = lambda: state["video_processor"].sample_frames()
    
    return { "sampled_generator_factory": sampled_generator_factory }


def remove_similar_frames(state: State):
    
    reduced_generator_factory = lambda: state["video_processor"].remove_similar_frames(state["sampled_generator_factory"]())
    
    return { "reduced_generator_factory": reduced_generator_factory }

def batch_frames(state: State):
    
    batched_generator_factory = lambda: state["video_processor"].generate_batches_of_frames(state["reduced_generator_factory"]())
    
    return { "batched_generator_factory": batched_generator_factory }

    
def expand_frame_range(state: State):
    
    def expanded_frame_range_generator_factory():
        matched_frame_range_generator = ( (start, end, None) for start, end in state["matched_frame_range"] )
    
        expanded_frame_range_generator = state["video_processor"].expand_frame_range(matched_frame_range_generator)
        
        return expanded_frame_range_generator
    
    return { "expanded_frame_range_generator_factory": expanded_frame_range_generator_factory }


def clip_video(state: State):
    
    state["video_processor"].create_video(state["expanded_frame_range_generator_factory"](), state["output_path"])
    
    return
    
    


