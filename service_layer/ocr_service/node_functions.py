import logging

from service_layer.ocr_service.state import State

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


def ensure_index(state: State):
    state["indexer"].ensure_index(
        video_path=state["video_path"],
        index_path=state["index_path"],
        meta_path=state["meta_path"],
        transcript_path=state["transcript_path"],
        batched_generator_factory=state.get("batched_generator_factory"),
        fps=state.get("fps"),
    )
    return {}


def query_timestamps(state: State):
    timestamps, search_mode = state["searcher"].query(
        regex_keywords=state["regex_keywords"],
        semantic_query=state["semantic_query"],
        index_path=state["index_path"],
        meta_path=state["meta_path"],
    )
    logging.info("OCR retrieval returned %s timestamp ranges.", len(timestamps))
    return {"timestamps": timestamps, "search_mode": search_mode}
