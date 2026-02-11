from router.clip_logic import clip_logic
from router.yolo_logic import yolo_logic

video_path = "video_storage/roses.mp4"

output_path = "outputs/output_yolo_rose_0.mp4"

user_prompt = "clips with one person with no roses"

# clip_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)
yolo_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)