import os

from config import settings
from infrastructure.ocr_indexer import OCR_Indexer
from infrastructure.ocr_searcher import OCR_Searcher
from router.main_state import Main_State
from service_layer.llm_service.graph import ocr_workflow as llm_ocr_workflow
from service_layer.llm_service.state import get_ocr_state as get_llm_ocr_state
from service_layer.ocr_service.graph import workflow as ocr_workflow
from service_layer.ocr_service.state import get_state as get_ocr_state


def ocr_logic(state: Main_State):
    video_path = state["video_path"]
    user_text = state["user_text"]
    video_state = state["video_state"]

    llm_ocr_state = get_llm_ocr_state(user_text)
    llm_ocr_state = llm_ocr_workflow.invoke(llm_ocr_state)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    index_dir = settings.OCR_INDEX_DIR
    index_path = os.path.join(index_dir, f"{video_name}.faiss")
    meta_path = os.path.join(index_dir, f"{video_name}.pkl")
    transcript_dir = settings.OCR_TRANSCRIPT_DIR
    transcript_path = os.path.join(transcript_dir, f"{video_name}.txt")

    ocr_state = get_ocr_state(
        video_path=video_path,
        index_path=index_path,
        meta_path=meta_path,
        transcript_path=transcript_path,
        regex_keywords=llm_ocr_state.get("regex_keywords", []),
        semantic_query=llm_ocr_state.get("semantic_query", ""),
        indexer=OCR_Indexer(),
        searcher=OCR_Searcher(),
        batched_generator_factory=video_state.get("batched_generator_factory"),
        fps=video_state["video_processor"].fps,
    )
    ocr_state = ocr_workflow.invoke(ocr_state)

    return {
        "timestamps": ocr_state["timestamps"],
        "route_details": {
            "path": "ocr",
            "regex_keywords": llm_ocr_state.get("regex_keywords", []),
            "semantic_query": llm_ocr_state.get("semantic_query", ""),
            "search_mode": ocr_state.get("search_mode", "none"),
            "timestamp_count": len(ocr_state.get("timestamps", [])),
        },
    }
