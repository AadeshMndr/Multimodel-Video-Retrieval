import logging
from statistics import mean, median
from typing import Callable

import numpy as np
import torch
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

from config import settings
from infrastructure.h5py_storage import Embedding_Store
from types_and_schemas.video_types import Generator_Image_Range


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class XCLIP_Processor:
    def __init__(
        self,
        model_name: str,
        embedding_store: Embedding_Store | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        requested_device = device or settings.DEVICE
        self.device = self._resolve_device(requested_device)
        self.embedding_store = embedding_store
        self.embedding_store_has_data = False

        if self.embedding_store is not None and settings.ENABLE_XCLIP_EMBEDDING_STORAGE:
            self.embedding_store_has_data = self.embedding_store.is_data_present()

        if self.device == "mps":
            import os
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        logging.info(f"Loading XCLIP model {self.model_name} on {self.device}...")
        self.model = XCLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model.config.return_dict = True
        self.model.vision_model.config.return_dict = True
        self.model.eval()
        logging.info("XCLIP model loaded successfully")

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if requested_device == "cpu":
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _fallback_to_cpu_if_needed(self, error: Exception):
        if self.device != "mps":
            raise error

        logging.warning(f"MPS operation failed for XCLIP, falling back to CPU: {error}")
        self.device = "cpu"
        self.model = self.model.to("cpu")

    def _get_text_embeddings(self, texts: list[str]) -> torch.Tensor:
        if len(texts) == 0:
            raise ValueError("XCLIP requires at least one text prompt")

        with torch.no_grad():
            try:
                text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
                text_features = self.model.get_text_features(**text_inputs)
            except Exception as error:
                self._fallback_to_cpu_if_needed(error)
                text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
                text_features = self.model.get_text_features(**text_inputs)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features

    def _get_video_embedding(self, frames: list[Image.Image]) -> torch.Tensor:
        with torch.no_grad():
            try:
                video_inputs = self.processor(images=frames, return_tensors="pt")
                if "pixel_values" not in video_inputs:
                    raise ValueError("XCLIP processor did not return 'pixel_values' for video frames")

                pixel_values = video_inputs["pixel_values"].to(self.device)
                if pixel_values.ndim != 5:
                    raise ValueError(f"Expected 5D pixel_values for video, got shape {tuple(pixel_values.shape)}")

                batch_size, num_frames, num_channels, height, width = pixel_values.shape
                flat_pixel_values = pixel_values.view(batch_size * num_frames, num_channels, height, width)

                vision_outputs = self.model.vision_model(pixel_values=flat_pixel_values, return_dict=True)
                frame_embeddings = vision_outputs.pooler_output.view(batch_size, num_frames, -1)
                video_embeddings = frame_embeddings.mean(dim=1)
                video_features = self.model.visual_projection(video_embeddings)
            except Exception as error:
                self._fallback_to_cpu_if_needed(error)
                video_inputs = self.processor(images=frames, return_tensors="pt")
                if "pixel_values" not in video_inputs:
                    raise ValueError("XCLIP processor did not return 'pixel_values' for video frames")

                pixel_values = video_inputs["pixel_values"].to(self.device)
                if pixel_values.ndim != 5:
                    raise ValueError(f"Expected 5D pixel_values for video, got shape {tuple(pixel_values.shape)}")

                batch_size, num_frames, num_channels, height, width = pixel_values.shape
                flat_pixel_values = pixel_values.view(batch_size * num_frames, num_channels, height, width)

                vision_outputs = self.model.vision_model(pixel_values=flat_pixel_values, return_dict=True)
                frame_embeddings = vision_outputs.pooler_output.view(batch_size, num_frames, -1)
                video_embeddings = frame_embeddings.mean(dim=1)
                video_features = self.model.visual_projection(video_embeddings)

            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            return video_features

    def _merge_ranges(self, frame_ranges: list[tuple[int, int]], fps: float) -> list[tuple[int, int]]:
        if len(frame_ranges) == 0:
            return []

        max_gap_frames = max(0, int(settings.XCLIP_MERGE_GAP_SECONDS * fps))
        sorted_ranges = sorted(frame_ranges, key=lambda frame_range: frame_range[0])

        merged: list[tuple[int, int]] = [sorted_ranges[0]]

        for current_start, current_end in sorted_ranges[1:]:
            previous_start, previous_end = merged[-1]
            if current_start <= previous_end + max_gap_frames:
                merged[-1] = (previous_start, max(previous_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def _get_window_params(self, fps: float) -> tuple[int, int]:
        sample_fps = fps / max(1, settings.VIDEO_SAMPLING_RATE)
        frames_per_window = max(
            settings.XCLIP_FRAMES_PER_CLIP,
            int(settings.XCLIP_WINDOW_SECONDS * sample_fps),
        )
        step_frames = max(1, int(settings.XCLIP_STEP_SECONDS * sample_fps))
        return frames_per_window, step_frames

    def _generate_window_frame_data(
        self,
        sampled_frames: list[tuple[int, int, Image.Image]],
        fps: float,
    ) -> tuple[list[list[Image.Image]], list[tuple[int, int]]]:
        frames_per_window, step_frames = self._get_window_params(fps)

        selected_window_frames: list[list[Image.Image]] = []
        frame_ranges: list[tuple[int, int]] = []

        for start_index in range(0, len(sampled_frames), step_frames):
            window_items = sampled_frames[start_index:start_index + frames_per_window]
            if len(window_items) < settings.XCLIP_FRAMES_PER_CLIP:
                break

            selection_indices = torch.linspace(
                0,
                len(window_items) - 1,
                steps=settings.XCLIP_FRAMES_PER_CLIP,
                dtype=torch.int64,
            ).tolist()

            selected_frames = [window_items[int(index)][2] for index in selection_indices]
            frame_start = window_items[0][0]
            frame_end = window_items[-1][1]

            selected_window_frames.append(selected_frames)
            frame_ranges.append((frame_start, frame_end))

        return selected_window_frames, frame_ranges

    def _load_stored_window_embeddings(self) -> tuple[torch.Tensor, list[tuple[int, int]]] | None:
        if self.embedding_store is None or not (settings.ENABLE_XCLIP_EMBEDDING_STORAGE and self.embedding_store_has_data):
            return None

        stored_embeddings, stored_ranges = self.embedding_store.get_all_embeddings()
        if len(stored_embeddings) == 0:
            return None

        frame_ranges = [
            (int(frame_range_data[0]), int(frame_range_data[1]))
            for frame_range_data in stored_ranges
        ]
        embeddings_tensor = torch.from_numpy(stored_embeddings).to(self.device)
        return embeddings_tensor, frame_ranges

    def _compute_and_optionally_store_window_embeddings(
        self,
        sampled_frames_factory: Callable[[], Generator_Image_Range],
        fps: float,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        sampled_frames = list(sampled_frames_factory())
        if len(sampled_frames) == 0:
            return torch.empty(0, 0, device=self.device), []

        selected_window_frames, frame_ranges = self._generate_window_frame_data(sampled_frames=sampled_frames, fps=fps)

        if len(selected_window_frames) == 0:
            return torch.empty(0, 0, device=self.device), []

        embedding_rows: list[torch.Tensor] = []
        for selected_frames in selected_window_frames:
            embedding_rows.append(self._get_video_embedding(selected_frames).squeeze(0).to("cpu"))

        embeddings_tensor = torch.stack(embedding_rows, dim=0).to(self.device)

        if self.embedding_store is not None and settings.ENABLE_XCLIP_EMBEDDING_STORAGE:
            embeddings_numpy = embeddings_tensor.to("cpu").numpy().astype("float32")
            ranges_numpy = np.array(frame_ranges, dtype="int")
            self.embedding_store.store_batch_embeddings(embeddings_numpy, ranges_numpy)
            self.embedding_store_has_data = True

        return embeddings_tensor, frame_ranges

    def _get_window_embeddings_and_ranges(
        self,
        sampled_frames_factory: Callable[[], Generator_Image_Range],
        fps: float,
    ) -> tuple[torch.Tensor, list[tuple[int, int]], bool]:
        stored_data = self._load_stored_window_embeddings()
        if stored_data is not None:
            embeddings_tensor, frame_ranges = stored_data
            return embeddings_tensor, frame_ranges, True

        embeddings_tensor, frame_ranges = self._compute_and_optionally_store_window_embeddings(
            sampled_frames_factory=sampled_frames_factory,
            fps=fps,
        )
        return embeddings_tensor, frame_ranges, False

    def find_temporal_matches(
        self,
        sampled_frames_factory: Callable[[], Generator_Image_Range],
        texts: list[str],
        fps: float,
    ) -> tuple[list[tuple[int, int]], dict]:
        frame_ranges, all_scores, score_stats = self.compute_window_scores(
            sampled_frames_factory=sampled_frames_factory,
            texts=texts,
            fps=fps,
        )

        matched_ranges: list[tuple[int, int]] = [
            frame_range
            for frame_range, score in zip(frame_ranges, all_scores)
            if score >= settings.XCLIP_THRESHOLD
        ]

        merged_ranges = self._merge_ranges(matched_ranges, fps=fps)
        return merged_ranges, score_stats

    def generate_and_store_window_embeddings(
        self,
        sampled_frames_factory: Callable[[], Generator_Image_Range],
        fps: float,
    ) -> dict:
        window_embeddings, frame_ranges, cache_hit = self._get_window_embeddings_and_ranges(
            sampled_frames_factory=sampled_frames_factory,
            fps=fps,
        )

        window_count = len(frame_ranges)
        embedding_count = int(window_embeddings.shape[0]) if window_embeddings.ndim >= 1 else 0

        return {
            "cache_hit": cache_hit,
            "window_count": window_count,
            "embedding_count": embedding_count,
        }

    def compute_window_scores(
        self,
        sampled_frames_factory: Callable[[], Generator_Image_Range],
        texts: list[str],
        fps: float,
    ) -> tuple[list[tuple[int, int]], list[float], dict]:
        prompt_texts = [text for text in texts if text.strip()]
        if len(prompt_texts) == 0:
            prompt_texts = [""]

        text_embeddings = self._get_text_embeddings(prompt_texts)

        window_embeddings, frame_ranges, cache_hit = self._get_window_embeddings_and_ranges(
            sampled_frames_factory=sampled_frames_factory,
            fps=fps,
        )

        if len(frame_ranges) == 0 or window_embeddings.numel() == 0:
            return [], [], {
                "window_count": 0,
                "matched_window_count": 0,
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0,
                "min": 0.0,
                "cache_hit": cache_hit,
                "used_threshold": settings.XCLIP_THRESHOLD,
            }

        if window_embeddings.device != text_embeddings.device or window_embeddings.dtype != text_embeddings.dtype:
            window_embeddings = window_embeddings.to(device=text_embeddings.device, dtype=text_embeddings.dtype)

        similarity_matrix = window_embeddings @ text_embeddings.T
        max_scores = similarity_matrix.max(dim=1).values
        all_scores = [float(score) for score in max_scores.tolist()]

        matched_ranges: list[tuple[int, int]] = [
            frame_range
            for frame_range, score in zip(frame_ranges, all_scores)
            if score >= settings.XCLIP_THRESHOLD
        ]

        matched_window_count = len(matched_ranges)
        window_count = len(frame_ranges)

        if len(all_scores) == 0:
            score_stats = {
                "window_count": 0,
                "matched_window_count": 0,
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0,
                "min": 0.0,
                "cache_hit": cache_hit,
                "used_threshold": settings.XCLIP_THRESHOLD,
            }
        else:
            score_stats = {
                "window_count": window_count,
                "matched_window_count": matched_window_count,
                "mean": float(mean(all_scores)),
                "median": float(median(all_scores)),
                "max": float(max(all_scores)),
                "min": float(min(all_scores)),
                "cache_hit": cache_hit,
                "used_threshold": settings.XCLIP_THRESHOLD,
            }

        return frame_ranges, all_scores, score_stats
