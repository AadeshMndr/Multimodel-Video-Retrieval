from router.logic_routes.clip_logic import clip_logic
from router.logic_routes.yolo_logic import yolo_logic

video_path = "video_storage/roses.mp4"

output_path = "outputs/output_yolo_rose_dummy.mp4"

user_prompt = "one roses and two person"

# clip_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)
yolo_logic(video_path=video_path, user_text=user_prompt, output_path=output_path)








# from infrastructure.video_processor import Video_Processor
# from infrastructure.clip_processor import CLIP_Processor


# video_path = "/home/aadesh-manandhar/home/projects/video_clipper/video_storage/home.mp4"

# output_path = "Output2.mp4"

# video_processor = Video_Processor(video_path=video_path)

# clip_processor = CLIP_Processor(model_name="ViT-L/14")


# sampled_frames = video_processor.sample_frames(sampling_rate=15)
# reduced_frames = video_processor.remove_similar_frames(sampled_frames)
# batch_generator = video_processor.generate_batches_of_frames(reduced_frames)

# list_of_frames, start_stop_data  = next(batch_generator)

# batch_of_frames = clip_processor.encode_frame_list(list_of_frames)

# text_output = clip_processor.encode_text_list([ "Person with blue jacket", "dog jumping" ])


# output = clip_processor.match_frames_and_text(batch_of_frames, text_output, start_stop_data)

# print(output)

# if len(output) > 0:

#     output_generator = ( (start, last, None) for start, last in output )

#     expanded_range = video_processor.expand_frame_range(output_generator)

#     video_processor.create_video(expanded_range, output_path=output_path)

# else:
    
#     print("No scenes matched")    