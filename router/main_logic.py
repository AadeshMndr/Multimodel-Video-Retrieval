from router.main_state import Main_State

from service_layer.video_service.state import State as VideoState
from service_layer.video_service.graph import pre_workflow as pre_processing, post_workflow as post_processing, refine_timestamps_workflow as timestamp_workflow
from infrastructure.video_processor import Video_Processor
from service_layer.video_service.state import get_state as get_video_state
from service_layer.llm_service.graph import analyzer_workflow
from service_layer.llm_service.state import get_analyzer_state
from config import settings
import logging
import threading
from threading import Event


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

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
    if state["logical_path_choosen"] == "ocr":
        batch_size = settings.OCR_BATCH_SIZE
    
    video_processor = Video_Processor(video_path=video_path)
    video_state = get_video_state(output_path=output_path, video_processor=video_processor, batch_size=batch_size)

    if state["logical_path_choosen"] != "audio":
        video_state = pre_processing.invoke(video_state)
    
    return { "video_state": video_state }


def postprocess(state: Main_State):
    
    video_state = state["video_state"]
    
    logging.info("=" * 60)
    logging.info("The frames that matched are: \n")
    for start, last in state["matched_frames"]:
        start_seconds = start / (video_state["video_processor"].fps + 0.0000001)
        last_seconds = last / (video_state["video_processor"].fps + 0.0000001)
        start_minutes = int(start_seconds / 60)
        last_minutes = int(last_seconds / 60)
        start_seconds = int(start_seconds) % 60
        last_seconds = int(last_seconds) % 60
        logging.info(f"Frame: {start}-{last} || Timestamp: {start_minutes}:{start_seconds}-{last_minutes}:{last_seconds}")
    logging.info("=" * 60)
    
    video_state["matched_frame_range"] = state["matched_frames"]
    
    post_processing.invoke(video_state) # type: ignore


def create_video_in_background(state: VideoState, video_creation_event: Event):
    try:
        post_processing.invoke(state)
    finally:
        video_creation_event.set()


def create_video_from_timestamps_in_background(video_state: VideoState, timestamps: list[tuple[int, int]], output_path: str, video_creation_event: Event):
    try:
        video_state["video_processor"].create_video_from_timestamps(timestamps=timestamps, output_path=output_path)
    finally:
        video_creation_event.set()


def parallel_post_process(state: Main_State):
    
    video_state = state["video_state"]
    should_create_video = state.get("generate_output_video", True)

    if state["logical_path_choosen"] in ("audio", "ocr"):
        timestamps = list(state.get("timestamps", []))
        if not should_create_video:
            return {"timestamps": timestamps, "video_state": video_state, "video_creation_event": None}

        video_creation_event = Event()
        video_thread = threading.Thread(
            target=create_video_from_timestamps_in_background,
            args=(video_state, timestamps, state["output_path"], video_creation_event),
        )
        video_thread.daemon = True
        video_thread.start()
        return {"timestamps": timestamps, "video_state": video_state, "video_creation_event": video_creation_event}
    
    video_state["matched_frame_range"] = state["matched_frames"]
    
    final_state: Main_State = timestamp_workflow.invoke(video_state) # type: ignore

    if not should_create_video:
        return {"timestamps": final_state["timestamps"], "video_state": video_state, "video_creation_event": None}
    
    video_creation_event = Event()
    
    # Let the video creation run in the background
    video_thread = threading.Thread(target=create_video_in_background, args=(video_state, video_creation_event))
    video_thread.daemon = True 
    video_thread.start()
    
    return { "timestamps": final_state["timestamps"], "video_state": video_state, "video_creation_event": video_creation_event }



    
  
