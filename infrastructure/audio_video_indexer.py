import logging
import os
import pickle
import time
from typing import Any

from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Audio_Video_Indexer:

    def __init__(
        self,
        whisper_model_size: str = settings.AUDIO_WHISPER_MODEL_SIZE,
        embedding_model: str = settings.AUDIO_EMBEDDING_MODEL,
    ):
        self.whisper_model_size = whisper_model_size
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
    def _compute_type_for_device(device: str) -> str:
        return "float16" if device == "cuda" else "int8"

    @staticmethod
    def _get_whisper_device(device: str) -> str:
        if device == "cuda":
            return "cuda"
        return "cpu"

    @staticmethod
    def _ensure_dependencies():
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Audio indexing dependencies are missing. Install "
                "'faster-whisper', 'sentence-transformers', and 'faiss-cpu'."
            ) from exc

    def transcribe(self, video_path: str) -> list[dict[str, Any]]:
        self._ensure_dependencies()
        from faster_whisper import WhisperModel

        whisper_device = self._get_whisper_device(self.device)
        if self.device != whisper_device:
            logging.info(
                "Audio transcription device '%s' is not supported by faster-whisper. Falling back to '%s'.",
                self.device,
                whisper_device,
            )

        model = WhisperModel(
            self.whisper_model_size,
            device=whisper_device,
            compute_type=self._compute_type_for_device(whisper_device),
        )

        logging.info("Transcribing audio for semantic indexing...")
        start_time = time.perf_counter()

        segments, _ = model.transcribe(
            video_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
        )

        transcript = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments
        ]

        logging.info(
            "Transcription complete in %.2fs (%s segments).",
            time.perf_counter() - start_time,
            len(transcript),
        )
        return transcript

    @staticmethod
    def chunk_by_time(
        transcript: list[dict[str, Any]],
        window_size: int = settings.AUDIO_CHUNK_WINDOW_SIZE,
        overlap_size: int = settings.AUDIO_CHUNK_OVERLAP_SIZE,
    ) -> list[dict[str, Any]]:
        if not transcript:
            return []

        chunks: list[dict[str, Any]] = []
        n = len(transcript)
        i = 0

        while i < n:
            current = [transcript[i]]
            chunk_start = transcript[i]["start"]
            j = i + 1

            while j < n:
                seg = transcript[j]
                current.append(seg)
                elapsed = seg["end"] - chunk_start

                if elapsed >= window_size:
                    ends_sentence = seg["text"].rstrip().endswith((".", "?", "!", "..."))
                    if ends_sentence or elapsed >= window_size + 10:
                        j += 1
                        break
                j += 1

            chunks.append(
                {
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                    "text": " ".join(s["text"].strip() for s in current),
                }
            )

            if overlap_size > 0 and len(current) > 1:
                overlap_start = current[-1]["end"] - overlap_size
                next_i = j - 1
                for k in range(j - 2, i, -1):
                    if transcript[k]["start"] >= overlap_start:
                        next_i = k
                    else:
                        break
                i = next_i
            else:
                i = j

        return chunks

    def build_and_save(self, chunks: list[dict[str, Any]], index_path: str, meta_path: str):
        self._ensure_dependencies()
        import faiss
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(self.embedding_model, device=self.device)
        texts = [chunk["text"] for chunk in chunks]
        prefixed_texts = [BGE_QUERY_PREFIX + text for text in texts]

        logging.info("Encoding %s audio chunks...", len(prefixed_texts))
        embeddings = embedder.encode(
            prefixed_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )

        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        index_dir = os.path.dirname(index_path)
        meta_dir = os.path.dirname(meta_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)

        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as file_obj:
            pickle.dump(chunks, file_obj)

    @staticmethod
    def should_reindex(video_path: str, index_path: str, meta_path: str) -> bool:
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return True

        video_mtime = os.path.getmtime(video_path)
        return video_mtime > os.path.getmtime(index_path) or video_mtime > os.path.getmtime(meta_path)

    def ensure_index(self, video_path: str, index_path: str, meta_path: str):
        if not settings.AUDIO_FORCE_REINDEX:
            if os.path.exists(index_path) and os.path.exists(meta_path):
                logging.info("Audio semantic index found. Reusing existing index.")
                return

        transcript = self.transcribe(video_path)
        chunks = self.chunk_by_time(transcript)

        if not chunks:
            logging.warning("No audio chunks were created. Writing an empty index is skipped.")
            return

        self.build_and_save(chunks, index_path=index_path, meta_path=meta_path)
        logging.info("Audio semantic index built successfully.")
