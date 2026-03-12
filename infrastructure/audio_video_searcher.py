import logging
import os
import pickle
from math import ceil

import numpy as np

from config import settings
from infrastructure.audio_video_indexer import BGE_QUERY_PREFIX

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


class Audio_Video_Searcher:

    def __init__(self, embedding_model: str = settings.AUDIO_EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.device = self._get_torch_device()

    @staticmethod
    def _get_torch_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon Mac GPU support
            else:
                return "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _ensure_dependencies():
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Audio search dependencies are missing. Install "
                "'sentence-transformers' and 'faiss-cpu'."
            ) from exc

    def query(
        self,
        user_query: str,
        index_path: str,
        meta_path: str,
        top_k: int = settings.AUDIO_TOP_K,
        merge_gap: float = settings.AUDIO_MERGE_GAP,
    ) -> list[tuple[int, int]]:
        self._ensure_dependencies()
        import faiss
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logging.warning("Audio index files were not found. Returning empty timestamps.")
            return []

        embedder = SentenceTransformer(self.embedding_model, device=self.device)
        index = faiss.read_index(index_path)

        with open(meta_path, "rb") as file_obj:
            documents: list[dict[str, float | str]] = pickle.load(file_obj)

        query_embedding = embedder.encode(
            [BGE_QUERY_PREFIX + user_query], convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, top_k)

        valid_scores = np.array([score for score in scores[0] if score != -1])
        if len(valid_scores) == 0:
            return []

        max_score = float(valid_scores.max())
        dynamic_threshold = max(max_score * 0.8, max_score - float(valid_scores.std()))

        raw_timestamps: list[tuple[float, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < dynamic_threshold:
                continue

            doc = documents[idx]
            start = float(doc["start"])
            end = float(doc["end"])
            raw_timestamps.append((start, end))

        merged = self.merge_adjacent(raw_timestamps, gap=merge_gap)
        return [
            (max(0, int(start)), max(0, ceil(end)))
            for start, end in merged
            if end >= start
        ]

    @staticmethod
    def merge_adjacent(
        ranges: list[tuple[float, float]], gap: float = settings.AUDIO_MERGE_GAP
    ) -> list[tuple[float, float]]:
        if not ranges:
            return []

        ranges = sorted(ranges, key=lambda each: each[0])
        merged = [list(ranges[0])]

        for curr_start, curr_end in ranges[1:]:
            _, prev_end = merged[-1]
            if curr_start - prev_end <= gap:
                merged[-1][1] = max(prev_end, curr_end)
            else:
                merged.append([curr_start, curr_end])

        return [(float(start), float(end)) for start, end in merged]
