import os

from config import settings
from infrastructure.audio_video_indexer import Audio_Video_Indexer
from infrastructure.audio_video_searcher import Audio_Video_Searcher
from router.main_state import Main_State
from service_layer.audio_service.graph import workflow as audio_workflow
from service_layer.audio_service.state import get_state as get_audio_state
from service_layer.llm_service.graph import audio_workflow as llm_audio_workflow
from service_layer.llm_service.state import get_audio_state as get_llm_audio_state


def audio_logic(state: Main_State):
    video_path = state["video_path"]
    user_text = state["user_text"]
    llm_audio_state = get_llm_audio_state(user_text)
    llm_audio_state = llm_audio_workflow.invoke(llm_audio_state)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    index_dir = settings.AUDIO_INDEX_DIR
    index_path = os.path.join(index_dir, f"{video_name}.faiss")
    meta_path = os.path.join(index_dir, f"{video_name}.pkl")

    audio_state = get_audio_state(
        video_path=video_path,
        user_query=llm_audio_state["refined_query"],
        index_path=index_path,
        meta_path=meta_path,
        indexer=Audio_Video_Indexer(),
        searcher=Audio_Video_Searcher(),
    )
    audio_state = audio_workflow.invoke(audio_state)

    return {
        "timestamps": audio_state["timestamps"],
        "route_details": {
            "path": "audio",
            "refined_query": llm_audio_state.get("refined_query", ""),
            "timestamp_count": len(audio_state.get("timestamps", [])),
        },
    }
