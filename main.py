from router.logic_routes.clip_logic import clip_logic
from router.logic_routes.yolo_logic import yolo_logic

video_path = "video_storage/roses.mp4"

output_path = "outputs/output_yolo_rose_dummy.mp4"

user_prompt = "one roses and two person"

# clip_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)
yolo_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)