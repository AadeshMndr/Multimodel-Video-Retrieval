import logging
import os
import pickle
import re
from math import ceil
from typing import Iterable

import numpy as np

from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


OCR_SUBSTITUTIONS = {
    "0": "[0Oo]",
    "1": "[1Il]",
    "2": "[2Z]",
    "5": "[5S]",
    "8": "[8B]",
    "o": "[0Oo]",
    "i": "[1Il]",
    "s": "[5S]",
}

SEP = r"[\s\-]+"


def _ocr_robust(token: str) -> str:
    return "".join(OCR_SUBSTITUTIONS.get(ch, re.escape(ch)) for ch in token)


def build_patterns(tokens: Iterable[str]) -> list[str]:
    token_list = [token.strip() for token in tokens if token.strip()]
    if not token_list:
        return []
    robust_tokens = [_ocr_robust(token) for token in token_list]
    seq_pattern = SEP.join(robust_tokens)
    lookaheads = "".join(rf"(?=.*\b{t}\b)" for t in robust_tokens)
    single_patterns = [rf"\b{t}\b" for t in robust_tokens] + robust_tokens
    return [seq_pattern, lookaheads] + single_patterns


def normalize_ocr_text(text: str) -> str:
    normalized = re.sub(r"(\d)\s+(\d)", r"\1-\2", text)
    return re.sub(r"\s+", " ", normalized).strip()


class OCR_Searcher:
    def __init__(self, embedding_model: str = settings.OCR_EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.device = self._get_torch_device()

    @staticmethod
    def _get_torch_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
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
                "OCR search dependencies are missing. Install "
                "'sentence-transformers' and 'faiss-cpu'."
            ) from exc

    def query(
        self,
        regex_keywords: Iterable[str],
        semantic_query: str,
        index_path: str,
        meta_path: str,
        top_k: int = settings.OCR_TOP_K,
        merge_gap: float = settings.OCR_MERGE_GAP,
    ) -> tuple[list[tuple[int, int]], str]:
        self._ensure_dependencies()

        regex_results: list[tuple[int, int]] = []
        patterns = build_patterns(regex_keywords)
        if patterns:
            regex_results = self._regex_search(patterns, meta_path, merge_gap)

        if regex_results:
            logging.info("OCR regex search matched %s ranges.", len(regex_results))
            return regex_results, "regex"

        if semantic_query:
            semantic_results = self._semantic_search(
                semantic_query, index_path, meta_path, top_k, merge_gap
            )
            logging.info("OCR semantic search matched %s ranges.", len(semantic_results))
            return semantic_results, "semantic"

        logging.info("OCR search returned no matches.")
        return [], "none"

    def _semantic_search(
        self,
        semantic_query: str,
        index_path: str,
        meta_path: str,
        top_k: int,
        merge_gap: float,
    ) -> list[tuple[int, int]]:
        import faiss
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logging.warning("OCR index files were not found. Returning empty timestamps.")
            return []

        embedder = SentenceTransformer(self.embedding_model, device=self.device)
        index = faiss.read_index(index_path)

        with open(meta_path, "rb") as file_obj:
            documents: list[dict[str, float | str]] = pickle.load(file_obj)

        query_embedding = embedder.encode(
            [settings.OCR_QUERY_PREFIX + semantic_query], convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, top_k)

        valid_scores_list = [
            float(score)
            for score, idx in zip(scores[0], indices[0])
            if idx != -1 and np.isfinite(score)
        ]
        if not valid_scores_list:
            return []

        max_score = max(valid_scores_list)
        if max_score < settings.OCR_MIN_SCORE:
            return []

        valid_scores = np.array(valid_scores_list, dtype=np.float32)
        dynamic_threshold = max(max_score * 0.8, max_score - float(valid_scores.std()))

        raw_timestamps: list[tuple[float, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < dynamic_threshold:
                continue
            doc = documents[idx]
            raw_timestamps.append((float(doc["start"]), float(doc["end"])))

        merged = self.merge_adjacent(raw_timestamps, gap=merge_gap)
        return [
            (max(0, int(start)), max(0, ceil(end)))
            for start, end in merged
            if end >= start
        ]

    def _regex_search(
        self,
        patterns: Iterable[str],
        meta_path: str,
        merge_gap: float,
    ) -> list[tuple[int, int]]:
        if not os.path.exists(meta_path):
            logging.warning("OCR metadata file was not found. Returning empty timestamps.")
            return []

        with open(meta_path, "rb") as file_obj:
            documents: list[dict[str, float | str]] = pickle.load(file_obj)

        matched: list[tuple[float, float]] = []
        for chunk in documents:
            text = normalize_ocr_text(str(chunk.get("text", "")))
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                matched.append((float(chunk["start"]), float(chunk["end"])))

        merged = self.merge_adjacent(matched, gap=merge_gap)
        return [
            (max(0, int(start)), max(0, ceil(end)))
            for start, end in merged
            if end >= start
        ]

    @staticmethod
    def merge_adjacent(
        ranges: list[tuple[float, float]], gap: float = settings.OCR_MERGE_GAP
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
