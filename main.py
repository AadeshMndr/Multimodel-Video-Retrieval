from router.main_state import get_main_state
from router.main_graph import main_workflow


video_path = "video_storage/cups.mp4"

output_path = "outputs/person_sitting.mp4"


user_prompt = "sitting down / being seated"


main_state = get_main_state(
    video_path=video_path,
    output_path=output_path,
    user_text=user_prompt
)

main_workflow.invoke(main_state)