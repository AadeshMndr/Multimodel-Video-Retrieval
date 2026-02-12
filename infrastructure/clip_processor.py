import clip
import logging
import torch
from torch import Tensor
from PIL import Image
from typing import Generator, Any
from config import settings
from types_and_schemas.video_types import Generator_Batch_Image_Range, Generator_Batch_Tensor_Range

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class CLIP_Processor:
    
    def __init__(self, model_name: str, device="cpu"):
        
        self.model_name = model_name
        self.device = device 
        
        logging.info(f"Loading model {self.model_name} in {self.device}...")
        
        self.model, self.preprocessor = clip.load(self.model_name, device=self.device)
        self.model.eval()
        
        logging.info(f"Model Loaded successfully!")
        

    def encode_text_list(self, text_list: list[str]) -> Tensor:
        
        with torch.no_grad():
            
            text_tokens = clip.tokenize(text_list).to(self.device)
            text_embeddings: Tensor = self.model.encode_text(text_tokens)
            
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            
        return text_embeddings
    
    def encode_frame_list(self, frame_list: list[Image.Image]) -> Tensor:
        
                
        with torch.no_grad():
            
            processed_frames: list[Tensor] = [ torch.as_tensor(self.preprocessor(frame)) for frame in frame_list ]
            frame_tensor = torch.stack(processed_frames, dim=0).to(self.device)
            
            frame_embeddings: Tensor = self.model.encode_image(frame_tensor)
        
            frame_embeddings /= frame_embeddings.norm(dim=-1, keepdim=True)
            
        return frame_embeddings
            
        
    def encode_frames(self, frame_generator: Generator_Batch_Image_Range) ->  Generator_Batch_Tensor_Range:
           
        for batch_of_frames, start_stop_data in frame_generator:
            
            yield (self.encode_frame_list(batch_of_frames), start_stop_data)
            
    
    def match_frames_and_text(self, batch_frame_embeddings: Tensor, batch_text_embeddings: Tensor, batch_frame_start_last: list[tuple[int, int]], threshold: float = settings.CLIP_THRESHOLD) -> tuple[list[tuple[int, int]], list[float]]:
        
        similarity_scores = batch_frame_embeddings @ batch_text_embeddings.T 

        max_scores = similarity_scores.max(dim=1).values
        matched_frames = max_scores >= threshold
        
        
        # matched_frames = similarity_scores >= threshold
        # matched_frames = matched_frames.any(dim=1)
        
        matched_frame_start_last = []
        
        for i, matched in enumerate(matched_frames.tolist()):
            if matched:
                matched_frame_start_last.append(batch_frame_start_last[i])
            
        return (matched_frame_start_last, max_scores.tolist())

        
    def get_only_matched_frames(self, embeddings_and_start_last_frame: Generator_Batch_Tensor_Range, text_embeddings: Tensor) -> tuple[list[tuple[int, int]], list[float]]:
        
        logging.info("Processing begins...")
        
        frames_scores: list[float] = []
        
        matched_frames_data: list[tuple[int, int]] = []
        
        # batch_number = 1
        
        for batch_frame_tensor, start_last_data in embeddings_and_start_last_frame:
            
            # logging.info("\n")
            # logging.info(f"Processing batch {batch_number}")
            # batch_number += 1
            
            matched_data, max_scores = self.match_frames_and_text(batch_frame_tensor, text_embeddings, start_last_data)
            
            # logging.info(f"Found {len(matched_data)} matching frame ranges")
            # logging.info("\n")
            matched_frames_data.extend(matched_data)
            frames_scores.extend(max_scores)
            
        logging.info(f"Number of Matched Frame Groups: {len(matched_frames_data)}")
            
        return (matched_frames_data, frames_scores)
            
            
        
        