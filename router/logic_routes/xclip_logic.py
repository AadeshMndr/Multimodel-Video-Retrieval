from config import settings
from infrastructure.h5py_storage import Embedding_Store
from infrastructure.xclip_processor import XCLIP_Processor
from router.main_state import Main_State
from service_layer.llm_service.graph import prompt_variation_workflow as llm_workflow
from service_layer.llm_service.state import get_modified_prompt_state as get_llm_state
from service_layer.xclip_service.graph import workflow as xclip_workflow
from service_layer.xclip_service.state import get_state as get_xclip_state


def xclip_logic(state: Main_State):
    user_text = state["user_text"]
    video_state = state["video_state"]
    video_path = state["video_path"]
    frame_source = "reduced" if settings.XCLIP_USE_REDUCED_FRAMES else "sampled"
    selected_frame_factory = video_state["reduced_generator_factory"] if settings.XCLIP_USE_REDUCED_FRAMES else video_state["sampled_generator_factory"]

    llm_state = get_llm_state(user_text)
    llm_state = llm_workflow.invoke(llm_state)

    xclip_store_dataset_name = (
        f"{video_path}__xclip__sr{settings.VIDEO_SAMPLING_RATE}"
        f"__w{settings.XCLIP_WINDOW_SECONDS}__step{settings.XCLIP_STEP_SECONDS}"
        f"__frames{settings.XCLIP_FRAMES_PER_CLIP}"
        f"__source_{frame_source}"
    )
    xclip_embedding_store = Embedding_Store(
        file_name=settings.XCLIP_EMBEDDING_STORE_FILEPATH,
        dataset_name=xclip_store_dataset_name,
        embedding_dimension=settings.XCLIP_EMBEDDING_DIMENSION,
        chunking_size=settings.XCLIP_EMBEDDING_CHUNK_SIZE,
    )

    try:
        xclip_processor = XCLIP_Processor(
            model_name=settings.XCLIP_MODEL_NAME,
            embedding_store=xclip_embedding_store,
            device=settings.DEVICE,
        )

        xclip_state = get_xclip_state(
            texts=llm_state.get("modified_prompts", []),
            sampled_frames_factory=selected_frame_factory,
            fps=video_state["video_processor"].fps,
            xclip_processor=xclip_processor,
        )
        xclip_state = xclip_workflow.invoke(xclip_state)

        matched_frames = xclip_state.get("matched_frames", [])
        score_stats = xclip_state.get("score_stats", {})

        return {
            "matched_frames": matched_frames,
            "route_details": {
                "path": "xclip",
                "modified_prompts": llm_state.get("modified_prompts", []),
                "score_stats": score_stats,
                "matched_frame_count": len(matched_frames),
                "temporal_analysis": True,
                "frame_source": frame_source,
            },
        }
    finally:
        xclip_embedding_store.close()


def generate_and_store_xclip_embeddings(state: Main_State):
    if not settings.CALCULATE_XCLIP_EMBEDDINGS_ON_UPLOAD:
        return

    video_state = state["video_state"]
    video_path = state["video_path"]
    frame_source = "reduced" if settings.XCLIP_USE_REDUCED_FRAMES else "sampled"
    selected_frame_factory = video_state["reduced_generator_factory"] if settings.XCLIP_USE_REDUCED_FRAMES else video_state["sampled_generator_factory"]

    xclip_store_dataset_name = (
        f"{video_path}__xclip__sr{settings.VIDEO_SAMPLING_RATE}"
        f"__w{settings.XCLIP_WINDOW_SECONDS}__step{settings.XCLIP_STEP_SECONDS}"
        f"__frames{settings.XCLIP_FRAMES_PER_CLIP}"
        f"__source_{frame_source}"
    )

    xclip_embedding_store = Embedding_Store(
        file_name=settings.XCLIP_EMBEDDING_STORE_FILEPATH,
        dataset_name=xclip_store_dataset_name,
        embedding_dimension=settings.XCLIP_EMBEDDING_DIMENSION,
        chunking_size=settings.XCLIP_EMBEDDING_CHUNK_SIZE,
    )

    try:
        xclip_processor = XCLIP_Processor(
            model_name=settings.XCLIP_MODEL_NAME,
            embedding_store=xclip_embedding_store,
            device=settings.DEVICE,
        )
        xclip_processor.generate_and_store_window_embeddings(
            sampled_frames_factory=selected_frame_factory,
            fps=video_state["video_processor"].fps,
        )
    finally:
        xclip_embedding_store.close()
