from router.main_state import get_main_state
from router.main_graph import main_workflow


video_path = "video_storage/bird_video.webm"

output_path = "outputs/white_eggs.mp4"


user_prompt = "eggs (Use Yolo)"


main_state = get_main_state(
    video_path=video_path,
    output_path=output_path,
    user_text=user_prompt
)

main_workflow.invoke(main_state)