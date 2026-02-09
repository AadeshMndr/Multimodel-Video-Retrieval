from router.clip_logic import clip_logic


video_path = "video_storage/bird_video.webm"

output_path = "outputs/output.mp4"

user_prompt = "ducks"

clip_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)