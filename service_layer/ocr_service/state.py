from typing import Callable, TypedDict

from infrastructure.ocr_indexer import OCR_Indexer
from infrastructure.ocr_searcher import OCR_Searcher
from types_and_schemas.video_types import Generator_Batch_Image_Range


class State(TypedDict):
    video_path: str
    index_path: str
    meta_path: str
    transcript_path: str
    regex_keywords: list[str]
    semantic_query: str
    batched_generator_factory: Callable[[], Generator_Batch_Image_Range] | None
    fps: float | None
    indexer: OCR_Indexer
    searcher: OCR_Searcher
    timestamps: list[tuple[int, int]]
    search_mode: str


def get_state(
    video_path: str,
    index_path: str,
    meta_path: str,
    transcript_path: str,
    regex_keywords: list[str],
    semantic_query: str,
    indexer: OCR_Indexer,
    searcher: OCR_Searcher,
    batched_generator_factory: Callable[[], Generator_Batch_Image_Range] | None = None,
    fps: float | None = None,
) -> State:
    return State(  # type: ignore
        video_path=video_path,
        index_path=index_path,
        meta_path=meta_path,
        transcript_path=transcript_path,
        regex_keywords=regex_keywords,
        semantic_query=semantic_query,
        batched_generator_factory=batched_generator_factory,
        fps=fps,
        indexer=indexer,
        searcher=searcher,
        timestamps=[],
        search_mode="none",
    )
