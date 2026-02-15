import json
from types_and_schemas.api_schemas import StreamOutput
from router.main_state import get_main_state, Main_State
from router.main_graph import main_workflow
from concurrent.futures import ProcessPoolExecutor
import asyncio

executor = ProcessPoolExecutor()

async def process_the_video(video_path: str, user_text: str, output_path: str, filename: str):
    
    stream_data = StreamOutput(status="initiating", timestamps=[])
    yield f"data: {json.dumps(stream_data)}"
    
    initial_state = get_main_state(
        video_path=video_path,
        user_text=user_text,
        output_path=output_path
    )
    
    stream_data = StreamOutput(status="processing", timestamps=[])
    yield f"data: {json.dumps(stream_data)}"
    
    loop = asyncio.get_event_loop()
    final_state: Main_State = await loop.run_in_executor( # type: ignore
        executor,
        main_workflow.invoke,
        initial_state
    )
    
    
    
    
    