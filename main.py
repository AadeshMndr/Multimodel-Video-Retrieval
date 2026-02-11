from router.main_state import Main_State, get_main_state
from router.main_graph import main_workflow


video_path = "video_storage/Spiderman.mp4"

output_path = "outputs/output_clip.mp4"


user_prompt = "find me clips of jet plane flying"


main_state = get_main_state(
    video_path=video_path,
    output_path=output_path,
    user_text=user_prompt
)

main_workflow.invoke(main_state)