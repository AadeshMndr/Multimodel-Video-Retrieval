from typing import TypedDict

from infrastructure.audio_video_indexer import Audio_Video_Indexer
from infrastructure.audio_video_searcher import Audio_Video_Searcher


class State(TypedDict):
    video_path: str
    user_query: str
    index_path: str
    meta_path: str
    indexer: Audio_Video_Indexer
    searcher: Audio_Video_Searcher
    timestamps: list[tuple[int, int]]


def get_state(
    video_path: str,
    user_query: str,
    index_path: str,
    meta_path: str,
    indexer: Audio_Video_Indexer,
    searcher: Audio_Video_Searcher,
) -> State:
    return State(  # type: ignore
        video_path=video_path,
        user_query=user_query,
        index_path=index_path,
        meta_path=meta_path,
        indexer=indexer,
        searcher=searcher,
        timestamps=[],
    )
