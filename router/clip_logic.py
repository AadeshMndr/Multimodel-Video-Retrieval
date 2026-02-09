from service_layer.video_service.graph import pre_workflow as pre_processing, post_workflow as post_processing
from service_layer.clip_service.graph import workflow as clip_workflow
from service_layer.clip_service.state import get_state as get_clip_state, State as CLIP_State
from service_layer.llm_service.state import get_state as get_llm_state 
from service_layer.llm_service.graph import workflow as llm_workflow
from infrastructure.video_processor import Video_Processor
from infrastructure.clip_processor import CLIP_Processor
from service_layer.video_service.state import get_state as get_video_state, State as Video_State
from config import settings

def clip_logic(video_path: str, user_text: str, output_path: str):
    
    video_processor = Video_Processor(video_path=video_path)
    video_state = get_video_state(output_path=output_path, video_processor=video_processor)
    video_state = pre_processing.invoke(video_state)
    
    llm_state = get_llm_state(user_text)
    llm_state = llm_workflow.invoke(llm_state)
    
    clip_processor = CLIP_Processor(model_name=settings.MODEL_NAME, device=settings.DEVICE)
    clip_state = get_clip_state(texts=llm_state["modified_prompts"], clip_processor=clip_processor, batch_frames_factory=video_state["batched_generator_factory"])
    clip_state = clip_workflow.invoke(clip_state)

    video_state["matched_frame_range"] = clip_state["matched_frames"]
    post_processing.invoke(video_state) # type: ignore
    
    
    
    
    
    
    