import cv2
import imagehash
from PIL import Image
from imagehash import ImageHash
import logging
import numpy as np
import os
import subprocess
import tempfile
from typing import Optional
from config import settings
from types_and_schemas.video_types import Generator_Generic_Range, Generator_Image_Range, Generator_Batch_Image_Range
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class Video_Processor:
    
    def __init__(self, video_path: str):
        
        self.video_path = video_path
        self.hamming_record = []
    
        cap = cv2.VideoCapture(video_path)
        
        self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        
        cap.release()
        
        logging.info("=" * 60)
        logging.info(f"Number of frames in Video: {self.frame_count}")
        logging.info(f"Video FPS: {self.fps}")
        logging.info(f"Frame_size: {self.frame_size}")
        if self.fps != 0:
            logging.info(f"Video Duration: {self.frame_count / self.fps} seconds")
        logging.info("=" * 60)
        logging.info(f"\n\n")
    
    
    def sample_frames(self, sampling_rate: int = settings.VIDEO_SAMPLING_RATE) -> Generator_Image_Range:
        
        cap = cv2.VideoCapture(self.video_path)
        
        frame_number = -1
        
        frames_kept = 0
       
        logging.info("\n") 
        pbar = tqdm(total=self.frame_count, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
        
        while True:
            
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_number += 1
            
            
            if frame_number % sampling_rate == 0:
                
                frames_kept += 1
                
                pbar.update(sampling_rate)
                
                pil_image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                
                yield (frame_number, frame_number, pil_image)
                
        cap.release()
        
        logging.info("=" * 60)
        logging.info(f"After Sampling frames:")
        logging.info(f"Number of frames kept: {frames_kept}")
        logging.info("=" * 60)
        logging.info("\n\n")
        
        
        
    def remove_similar_frames(self, frames: Generator_Image_Range, threshold: float = settings.PHASH_SIMILARITY_THRESHOLD) -> Generator_Image_Range:
        
        last_frame_number, _, last_frame = next(frames)
        last_frame_hash = imagehash.phash(last_frame)
        
        last_frame_number_seen = 0
        frames_kept = 0
        
        for frame_number, _, frame in frames:
        
            current_hash = imagehash.phash(frame)
                
            last_frame_number_seen = frame_number
                
            if self._two_frames_are_similar_by_hash(current_hash, last_frame_hash, threshold):    
                continue
                
            # logging.info(f"The frames {last_frame_number} to {frame_number} : Using {last_frame_number}")
                
            frames_kept += 1
                
            yield (last_frame_number, last_frame_number_seen - 1, last_frame)
                
            last_frame = frame
            last_frame_number = frame_number
            last_frame_hash = current_hash
        
     
        frames_kept += 1
        yield (last_frame_number, last_frame_number_seen, last_frame)
            

        logging.info("=" * 60)
        logging.info(f"After Reducing similar frames:")
        logging.info(f"Number of frames kept: {frames_kept}")
        logging.info("=" * 60)
        logging.info("\n\n")          
                
    def _two_frames_are_similar_by_hash(self, hash_a: ImageHash, hash_b: ImageHash, threshold):
        
        hamming_distance = hash_a - hash_b
        
        # logging.info(f"hamming distance: {hamming_distance}")
        
        self.hamming_record.append(hamming_distance)
        
        return hamming_distance <= threshold
    
    
    def create_video_of_just_frames(self, frames: Generator_Image_Range, output_path: str):
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(filename=output_path, fps=self.fps, fourcc=fourcc, frameSize=self.frame_size)
        
        for start_frame_num, end_frame_number, frame in frames:
            
            cv_image = np.array(frame)
            
            cv_frame = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            out.write(cv_frame)
        
        out.release()
        
    def expand_frame_range(self, frames: Generator_Generic_Range, frame_neighbour_range=(settings.FRAME_NEIGHBOUR_RANGE_BEFORE, settings.FRAME_NEIGHBOUR_RANGE_AFTER)) -> Generator_Generic_Range:   
        
        start, last, _ = next(frames, (None, None, None))
        
        if start is None or last is None:
            logging.info("There were no matched frames")
            return
        
        current_start = max(start - frame_neighbour_range[0], 0)
        current_last = min(last + frame_neighbour_range[1], self.frame_count - 1)
        
        for start_frame, last_frame, _ in frames:
            
            expanded_start = max(start_frame - frame_neighbour_range[0], 0)
            expanded_end = min(last_frame + frame_neighbour_range[1], self.frame_count - 1)
            
            # logging.info(f"expand frame: {start_frame}-{last_frame}")
            
            if expanded_start <= current_last:
                current_last = expanded_end
            else:
                yield (current_start, current_last, None)
            
                current_start = expanded_start
                current_last = expanded_end
                
        yield (current_start, current_last, None)
            
        
    def create_video(
        self,
        frames: Generator_Generic_Range,
        output_path: str,
        audio_timestamps: Optional[list[tuple[int, int]]] = None,
    ):
        
        cap = cv2.VideoCapture(filename=self.video_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output_path = temp_file.name
        
        out = cv2.VideoWriter(filename=temp_output_path, fps=self.fps, fourcc=fourcc, frameSize=self.frame_size)
        
        logging.info("Creating the video...")
        
        for start_frame, end_frame, _ in frames:
            
            logging.info(f"Adding frames: {start_frame}-{end_frame}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            
            while current_frame <= end_frame:
                ret, frame = cap.read()
                
                if ret:
                    out.write(frame)
                    
                current_frame += 1

        logging.info("Video Creation complete.")
                
        out.release()
        cap.release()
        
        self._mux_audio_with_ffmpeg(
            temp_output_path=temp_output_path,
            final_output_path=output_path,
            audio_timestamps=audio_timestamps,
        )

    def _build_audio_segments_file(self, timestamps: list[tuple[int, int]]) -> Optional[str]:
        valid_timestamps = [
            (max(0, int(start_seconds)), max(0, int(end_seconds)))
            for start_seconds, end_seconds in timestamps
            if int(end_seconds) > int(start_seconds)
        ]
        if not valid_timestamps:
            return None

        filter_parts = []
        concat_inputs = []
        for idx, (start_seconds, end_seconds) in enumerate(valid_timestamps):
            filter_parts.append(
                f"[0:a]atrim=start={start_seconds}:end={end_seconds},asetpts=PTS-STARTPTS[a{idx}]"
            )
            concat_inputs.append(f"[a{idx}]")

        concat_filter = "".join(concat_inputs) + f"concat=n={len(valid_timestamps)}:v=0:a=1[outa]"
        filter_complex = ";".join(filter_parts + [concat_filter])

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        extract_audio_command = [
            "ffmpeg",
            "-y",
            "-i",
            self.video_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-c:a",
            "aac",
            temp_audio_path,
        ]

        try:
            subprocess.run(
                extract_audio_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return temp_audio_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass
            return None

    def _mux_audio_with_ffmpeg(
        self,
        temp_output_path: str,
        final_output_path: str,
        audio_timestamps: Optional[list[tuple[int, int]]] = None,
    ):
        temp_audio_path = None
        if audio_timestamps:
            temp_audio_path = self._build_audio_segments_file(audio_timestamps)

        audio_input = temp_audio_path if temp_audio_path is not None else self.video_path
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output_path,
            "-i",
            audio_input,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-shortest",
            final_output_path,
        ]

        try:
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.warning("Could not mux audio with ffmpeg, falling back to silent video output.")
            os.replace(temp_output_path, final_output_path)
            if temp_audio_path is not None:
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass
            return

        try:
            os.remove(temp_output_path)
        except OSError:
            pass

        if temp_audio_path is not None:
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass

    def create_video_from_timestamps(self, timestamps: list[tuple[int, int]], output_path: str):
        def frames_generator():
            for start_seconds, end_seconds in timestamps:
                start_frame = max(0, int(start_seconds * self.fps))
                end_frame = min(self.frame_count - 1, int(end_seconds * self.fps))
                if end_frame >= start_frame:
                    yield (start_frame, end_frame, None)

        self.create_video(frames_generator(), output_path, audio_timestamps=timestamps)
           
    def get_timestamps(self, frames: Generator_Generic_Range):
        
        timestamps: list[tuple[int, int]] = []
        
        for start_frame, end_frame, _ in frames:
            
            epsilon_plus_fps = (self.fps + 0.000000001)
            start_seconds = int(start_frame / epsilon_plus_fps)
            end_seconds = int(end_frame / epsilon_plus_fps)
            
            timestamps.append((start_seconds, end_seconds))
            
        return timestamps
             
            
    def generate_batches_of_frames(self, frames: Generator_Image_Range, batch_size=settings.FRAME_BATCH_SIZE) -> Generator_Batch_Image_Range:
        
        frame_batch = []
        frame_start_last_numbers: list[tuple[int, int]] = []
        
        for start, last, frame in frames:
            
            if len(frame_batch) < batch_size:
                
                frame_batch.append(frame)
                frame_start_last_numbers.append((start, last))
                
            else:
                
                yield frame_batch, frame_start_last_numbers
                
                frame_batch = []
                frame_start_last_numbers = []
                
        if len(frame_batch) != 0:
            
            yield frame_batch, frame_start_last_numbers

            
            
           


if __name__ == "__main__":
    video_path = "video_storage/home.mp4"
    output_path = "Output.mp4"
    
    video_processor = Video_Processor(video_path=video_path)
    
    sampled_frames = video_processor.sample_frames(sampling_rate=15)
    removed_similar_frames = video_processor.remove_similar_frames(sampled_frames)
    expanded_frame_range = video_processor.expand_frame_range(removed_similar_frames)
    video_processor.create_video(expanded_frame_range, output_path)
    # video_processor.create_video_of_just_frames(removed_similar_frames, output_path)
    
    new_video_processor = Video_Processor(video_path=output_path)
    
    
    # print(f"Mean: {sum(video_processor.hamming_record) / len(video_processor.hamming_record)}")
    # print(f"Max: {max(video_processor.hamming_record)}, Min: {min(video_processor.hamming_record)}")
