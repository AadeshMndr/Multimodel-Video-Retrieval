import logging

from service_layer.audio_service.state import State

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


def ensure_index(state: State):
    state["indexer"].ensure_index(
        video_path=state["video_path"],
        index_path=state["index_path"],
        meta_path=state["meta_path"],
    )
    return {}


def query_timestamps(state: State):
    timestamps = state["searcher"].query(
        user_query=state["user_query"],
        index_path=state["index_path"],
        meta_path=state["meta_path"],
    )
    logging.info("Audio retrieval returned %s timestamp ranges.", len(timestamps))
    return {"timestamps": timestamps}
