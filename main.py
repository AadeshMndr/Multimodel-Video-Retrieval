from router.clip_logic import clip_logic

video_path = "video_storage/football_match.mp4"

output_path = "outputs/output.mp4"

user_prompt = "Clips of penalty shootout"

clip_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)