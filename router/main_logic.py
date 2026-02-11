from router.main_state import Main_State

from service_layer.video_service.graph import pre_workflow as pre_processing, post_workflow as post_processing
from infrastructure.video_processor import Video_Processor
from service_layer.video_service.state import get_state as get_video_state
from service_layer.llm_service.graph import analyzer_workflow
from service_layer.llm_service.state import get_analyzer_state
from config import settings

def prompt_analyzer(state: Main_State):
    
    user_text = state["user_text"]
    
    analyzer_state = get_analyzer_state(user_text)
    analyzer_state = analyzer_workflow.invoke(analyzer_state)
    
    return { "logical_path_choosen": analyzer_state["logical_path"]}

def decide_route(state: Main_State):
    
    return state["logical_path_choosen"]


def preprocess(state: Main_State):
    
    video_path = state["video_path"]
    output_path = state["output_path"]
    
    batch_size=settings.FRAME_BATCH_SIZE
    
    if state["logical_path_choosen"] == "yolo":
        batch_size = settings.YOLO_BATCH_SIZE
    
    video_processor = Video_Processor(video_path=video_path)
    video_state = get_video_state(output_path=output_path, video_processor=video_processor, batch_size=batch_size)
    video_state = pre_processing.invoke(video_state)
    
    return { "video_state": video_state }


def postprocess(state: Main_State):
    
    video_state = state["video_state"]
    
    video_state["matched_frame_range"] = state["matched_frames"]
    
    post_processing.invoke(video_state) # type: ignore