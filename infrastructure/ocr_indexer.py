import logging
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import settings
from infrastructure.video_processor import Video_Processor
from types_and_schemas.video_types import Generator_Batch_Image_Range, Generator_Image_Range

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


class OCR_Indexer:
    def __init__(
        self,
        text_detector_model: str = settings.OCR_TEXT_DETECTOR_MODEL,
        embedding_model: str = settings.OCR_EMBEDDING_MODEL,
        batch_size: int = settings.OCR_BATCH_SIZE,
        device: str = settings.DEVICE,
    ):
        self.text_detector_model = text_detector_model
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = device
        self._text_detector = None
        self._ocr_reader = None

    @staticmethod
    def _ensure_dependencies():
        try:
            import easyocr  # noqa: F401
            from ultralytics import YOLO  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            import faiss  # noqa: F401
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OCR indexing dependencies are missing. Install "
                "'easyocr', 'ultralytics', 'sentence-transformers', and 'faiss-cpu'."
            ) from exc

    def _ensure_models(self):
        if self._text_detector is not None and self._ocr_reader is not None:
            return

        from ultralytics import YOLO
        import easyocr

        self._text_detector = YOLO(model=self.text_detector_model)
        self._ocr_reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"))

    @staticmethod
    def _default_frame_generator(video_processor: Video_Processor) -> Generator_Image_Range:
        sampled = video_processor.sample_frames()
        reduced = video_processor.remove_similar_frames(sampled)
        return reduced

    def _batch_generator(
        self,
        frame_generator: Generator_Image_Range,
        video_processor: Video_Processor,
        batch_size: int,
    ) -> Generator_Batch_Image_Range:
        return video_processor.generate_batches_of_frames(frame_generator, batch_size=batch_size)

    def _filter_text_frames(
        self,
        batch_frames: list,
        start_last_data: list[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], np.ndarray]]:
        import cv2

        if not batch_frames:
            return []

        frames_bgr = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in batch_frames]

        results = self._text_detector.predict(  # type: ignore[union-attr]
            source=frames_bgr,
            batch=len(frames_bgr),
            conf=settings.OCR_TEXT_DETECTOR_CONF,
            device=self.device,
            verbose=False,
        )

        filtered: list[tuple[tuple[int, int], np.ndarray]] = []
        for result, frame_bgr, frame_range in zip(results, frames_bgr, start_last_data):
            if result.boxes is not None and len(result.boxes) > 0:
                filtered.append((frame_range, frame_bgr))
        return filtered

    def _run_ocr(self, frames_bgr: list[np.ndarray]) -> list[list[str]]:
        if not frames_bgr:
            return []
        import cv2

        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        return self._ocr_reader.readtext_batched(frames_rgb, detail=0)  # type: ignore[union-attr]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(ch.lower() for ch in text if ch.isalnum())

    def _build_chunks(
        self,
        batched_generator_factory: Callable[[], Generator_Batch_Image_Range],
        fps: float,
    ) -> tuple[list[dict[str, float | str]], list[str]]:
        self._ensure_models()

        chunks: list[dict[str, float | str]] = []
        transcript_lines: list[str] = []
        last_norm = ""
        last_end_seconds: float | None = None

        logging.info("Detecting text regions and running OCR...")
        start_time = time.perf_counter()

        for batch_frames, start_last_data in batched_generator_factory():
            filtered = self._filter_text_frames(batch_frames, start_last_data)
            if not filtered:
                continue

            frame_ranges, frames_bgr = zip(*filtered)
            ocr_texts = self._run_ocr(list(frames_bgr))

            for (start_frame, end_frame), text_list in zip(frame_ranges, ocr_texts):
                text = " ".join(text_list).strip()
                if not text:
                    continue

                normalized = self._normalize_text(text)
                if (
                    normalized == last_norm
                    and last_end_seconds is not None
                    and (start_seconds - last_end_seconds) <= settings.OCR_MERGE_GAP
                ):
                    continue

                start_seconds = float(start_frame / (fps + 1e-9))
                end_seconds = float(end_frame / (fps + 1e-9))

                chunks.append(
                    {
                        "start": start_seconds,
                        "end": max(start_seconds, end_seconds),
                        "text": text,
                    }
                )

                time_str = time.strftime("%H:%M:%S", time.gmtime(start_seconds))
                transcript_lines.append(f"[{time_str}] {text}")
                last_norm = normalized
                last_end_seconds = max(start_seconds, end_seconds)

        logging.info(
            "OCR pass complete in %.2fs (%s chunks).",
            time.perf_counter() - start_time,
            len(chunks),
        )
        return chunks, transcript_lines

    def _build_and_save_index(self, chunks: list[dict[str, float | str]], index_path: str, meta_path: str):
        import faiss
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(self.embedding_model, device=self.device)
        texts = [settings.OCR_QUERY_PREFIX + str(chunk["text"]) for chunk in chunks]

        logging.info("Encoding %s OCR chunks...", len(texts))
        embeddings = embedder.encode(
            texts,
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

    def ensure_index(
        self,
        video_path: str,
        index_path: str,
        meta_path: str,
        transcript_path: Optional[str] = None,
        batched_generator_factory: Optional[Callable[[], Generator_Batch_Image_Range]] = None,
        fps: Optional[float] = None,
    ):
        self._ensure_dependencies()

        if not settings.OCR_FORCE_REINDEX and not self.should_reindex(video_path, index_path, meta_path):
            logging.info("OCR index found. Reusing existing index.")
            return

        if batched_generator_factory is None or fps is None:
            video_processor = Video_Processor(video_path=video_path)
            batched_generator_factory = lambda: self._batch_generator(
                self._default_frame_generator(video_processor),
                video_processor,
                self.batch_size,
            )
            fps = video_processor.fps

        chunks, transcript_lines = self._build_chunks(batched_generator_factory, fps)

        if settings.OCR_WRITE_TRANSCRIPT and transcript_path:
            Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
            Path(transcript_path).write_text("\n".join(transcript_lines), encoding="utf-8")
            logging.info("OCR transcript saved to %s", transcript_path)

        if not chunks:
            logging.warning("No OCR chunks were created. Skipping index build.")
            return

        self._build_and_save_index(chunks, index_path=index_path, meta_path=meta_path)
        logging.info("OCR semantic index built successfully.")
