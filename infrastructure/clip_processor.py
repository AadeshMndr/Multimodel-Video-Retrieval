import clip
import logging
import torch
from torch import Tensor
from PIL import Image
from config import settings
from types_and_schemas.video_types import Generator_Batch_Image_Range, Generator_Batch_Tensor_Range
from types_and_schemas.generic_detection_types import ScoreData
from infrastructure.h5py_storage import Embedding_Store
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class CLIP_Processor:
    
    def __init__(self, model_name: str,  embedding_store: Embedding_Store, device="cpu"):
        
        self.model_name = model_name
        self.device = device 
        self.embedding_store = embedding_store
        self.embedding_store_has_data: bool = self.embedding_store.is_data_present() if settings.ENABLE_EMBEDDING_STORAGE else False
        
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
        
        if settings.ENABLE_EMBEDDING_STORAGE and self.embedding_store_has_data:

            logging.info("Using stored embeddings...")
            
            for batch_of_frames, start_last_data in self.embedding_store.generate_batch_embeddings(settings.FRAME_BATCH_SIZE):
                
                batch_frames_tensor = torch.from_numpy(batch_of_frames).to(settings.DEVICE)
                
                start_last_tuples = [ tuple(row) for row in start_last_data ]
                
                yield (batch_frames_tensor, start_last_tuples)

        else:                
           
            for batch_of_frames, start_last_data in frame_generator:
                
                batch_frame_encodings = self.encode_frame_list(batch_of_frames)

                if settings.CALCULATE_EMBEDDINGS_ON_PROCESSING:
                
                    batch_frames_numpy = batch_frame_encodings.to("cpu").numpy()
                    start_last_data_numpy = np.array(start_last_data)
                    
                    self.embedding_store.store_batch_embeddings(batch_frames_numpy, start_last_data_numpy)
                
                yield (batch_frame_encodings, start_last_data)
                
            if settings.CALCULATE_EMBEDDINGS_ON_PROCESSING:    
                self.embedding_store_has_data = True
            
    
    def match_frames_and_text(self, batch_frame_embeddings: Tensor, batch_text_embeddings: Tensor, batch_frame_start_last: list[tuple[int, int]], threshold: float = settings.CLIP_THRESHOLD) -> tuple[list[tuple[int, int]], list[ScoreData]]:
        if (
            batch_frame_embeddings.dtype != batch_text_embeddings.dtype
            or batch_frame_embeddings.device != batch_text_embeddings.device
        ):
            batch_text_embeddings = batch_text_embeddings.to(
                device=batch_frame_embeddings.device,
                dtype=batch_frame_embeddings.dtype,
            )

        similarity_scores = batch_frame_embeddings @ batch_text_embeddings.T 

        max_scores = similarity_scores.max(dim=1).values
        matched_frames = max_scores >= threshold
        
        max_scores_list = max_scores.tolist()
        
        score_data: list[ScoreData] = []
        
        # matched_frames = similarity_scores >= threshold
        # matched_frames = matched_frames.any(dim=1)
        
        matched_frame_start_last = []
        
        for i, matched in enumerate(matched_frames.tolist()):
            if matched:
                matched_frame_start_last.append(batch_frame_start_last[i])
            
            score_data.append((batch_frame_start_last[i], max_scores_list[i]))
            
        return (matched_frame_start_last, score_data)

        
    def get_only_matched_frames(self, embeddings_and_start_last_frame: Generator_Batch_Tensor_Range, text_embeddings: Tensor) -> tuple[list[tuple[int, int]], list[ScoreData]]:
        
        logging.info("Processing begins...")
        
        frames_scores: list[ScoreData] = []
        
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
            frames_scores.extend((max_scores))
            
        logging.info(f"Number of Matched Frame Groups: {len(matched_frames_data)}")
            
        return (matched_frames_data, frames_scores)
            
            
        
        
