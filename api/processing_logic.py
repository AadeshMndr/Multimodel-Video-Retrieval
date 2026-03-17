from types_and_schemas.api_schemas import StreamOutput
from router.main_state import get_main_state, Main_State
from router.main_graph import main_workflow
import asyncio

async def process_the_video(video_path: str, user_text: str, output_path: str, generate_output_video: bool = True):
    
    stream_data = StreamOutput(status="initiating", timestamps=[])
    yield f"data: {stream_data.model_dump_json()} \n\n"
    
    initial_state = get_main_state(
        video_path=video_path,
        user_text=user_text,
        output_path=output_path,
        generate_output_video=generate_output_video,
    )
    
    stream_data = StreamOutput(status="processing", timestamps=[])
    yield f"data: {stream_data.model_dump_json()} \n\n"
    
    # loop = asyncio.get_event_loop()
    # final_state: Main_State = await loop.run_in_executor( # type: ignore
    #     executor,
    #     main_workflow.invoke,
    #     initial_state
    # )
    
    final_state: Main_State = await asyncio.to_thread( # type: ignore
        main_workflow.invoke,
        initial_state
    )
    
    stream_data = StreamOutput(status="creating_video", timestamps=final_state["timestamps"])
    yield f"data: {stream_data.model_dump_json()} \n\n"
    
    if final_state["video_creation_event"] is not None:
        await asyncio.to_thread(final_state["video_creation_event"].wait)
        
        
    stream_data = StreamOutput(status="complete", timestamps=final_state["timestamps"])
    yield f"data: {stream_data.model_dump_json()} \n\n"
    
    
     
    