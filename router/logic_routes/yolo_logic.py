from service_layer.yolo_service.graph import workflow as yolo_workflow
from service_layer.yolo_service.state import get_state as get_yolo_state
from service_layer.video_service.graph import pre_workflow as pre_processing, post_workflow as post_processing
from service_layer.llm_service.graph import yolo_workflow as llm_for_yolo_workflow
from service_layer.llm_service.state import get_yolo_state as get_llm_for_yolo_state, YOLO_State as LLM_For_YOLO_State
from infrastructure.video_processor import Video_Processor
from service_layer.video_service.state import get_state as get_video_state
from infrastructure.yolo_processor import YOLO_Processor
from config import settings
from router.main_state import Main_State

def yolo_logic(state: Main_State):
    
    # video_processor = Video_Processor(video_path=video_path)
    # video_state = get_video_state(output_path=output_path, video_processor=video_processor, batch_size=settings.YOLO_BATCH_SIZE)
    # video_state = pre_processing.invoke(video_state)
    
    user_text = state["user_text"]
    video_state = state["video_state"]
   
    llm_state = get_llm_for_yolo_state(user_text)
    llm_state = llm_for_yolo_workflow.invoke(llm_state) # type: ignore
    
    yolo_processor = YOLO_Processor()
    
    yolo_state = get_yolo_state(
        yolo_processor=yolo_processor,
        batch_frames_factory=video_state["batched_generator_factory"],
        canonical_form=llm_state["canonical_form"],
        object_details=llm_state["object_details"],
        object_groups=llm_state["object_groups"]
    )
    yolo_state = yolo_workflow.invoke(yolo_state)
   
    return {
        "matched_frames": yolo_state["matched_frames"],
        "route_details": {
            "path": "yolo",
            "canonical_form": llm_state.get("canonical_form"),
            "object_groups": llm_state.get("object_groups", []),
            "object_details": llm_state.get("object_details", {}),
            "score_stats": yolo_state.get("score_stats", {}),
            "reassessment_count": yolo_state.get("reassessment_count", 0),
            "matched_frame_count": len(yolo_state.get("matched_frames", [])),
        },
    }
    
    # video_state["matched_frame_range"] = yolo_state["matched_frames"]
    # post_processing.invoke(video_state) # type: ignore
    
    