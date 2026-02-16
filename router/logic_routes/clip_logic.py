from service_layer.clip_service.graph import workflow as clip_workflow
from service_layer.clip_service.state import get_state as get_clip_state
from service_layer.llm_service.state import get_modified_prompt_state as get_llm_state 
from service_layer.llm_service.graph import prompt_variation_workflow as llm_workflow
from service_layer.clip_service.graph import generate_and_store_embeddings_workflow
from infrastructure.clip_processor import CLIP_Processor
from config import settings
from router.main_state import Main_State


def clip_logic(state: Main_State):
    
    user_text = state["user_text"]
    video_state = state["video_state"]
    
    llm_state = get_llm_state(user_text)
    llm_state = llm_workflow.invoke(llm_state)
    
    clip_processor = CLIP_Processor(model_name=settings.MODEL_NAME, embedding_store=state["embedding_store"], device=settings.DEVICE)
    clip_state = get_clip_state(texts=llm_state["modified_prompts"], clip_processor=clip_processor, batch_frames_factory=video_state["batched_generator_factory"])
    clip_state = clip_workflow.invoke(clip_state)
    
    return {"matched_frames": clip_state["matched_frames"]}
    
    
def generate_and_store_embeddings(state: Main_State):
    
    video_state = state["video_state"]
    
    clip_processor = CLIP_Processor(model_name=settings.MODEL_NAME, embedding_store=state["embedding_store"], device=settings.DEVICE)
    clip_state = get_clip_state(texts=[], clip_processor=clip_processor, batch_frames_factory=video_state["batched_generator_factory"]) 
    
    generate_and_store_embeddings_workflow.invoke(clip_state)
    
    