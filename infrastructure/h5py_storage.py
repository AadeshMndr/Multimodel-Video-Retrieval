import h5py
import numpy as np
from numpy.typing import NDArray
from typing import Generator
from h5py import Dataset
import os
import logging
from datetime import datetime


class Embedding_Store:
    
    def __init__(self, file_name: str, dataset_name: str, embedding_dimension: int, chunking_size: int):
        
        self.embedding_dimension = embedding_dimension
        self.file_name = f"{file_name}.h5"
        self.dataset_name = dataset_name
        self.chunking_size = chunking_size
        
        self.f = self._open_file_with_recovery(self.file_name)
        
        group = self.f.require_group(dataset_name)
            
        if "embeddings" not in group:
            
            group.create_dataset(                  
                "embeddings",
                shape=(0, self.embedding_dimension),
                maxshape=(None, self.embedding_dimension),
                chunks=(self.chunking_size, self.embedding_dimension),
                compression="gzip",
                dtype="float32"
            )
            
        if "start_last_data" not in group:
            
            group.create_dataset(
                "start_last_data",
                shape=(0, 2),
                maxshape=(None, 2),
                chunks=(self.chunking_size, 2),
                compression="gzip",
                dtype="int"
            )

    def _open_file_with_recovery(self, file_name: str) -> h5py.File:
        try:
            return h5py.File(file_name, "a")
        except OSError as error:
            error_message = str(error).lower()
            is_recoverable = (
                "truncated file" in error_message
                or "unable to synchronously open file" in error_message
                or "file signature not found" in error_message
            )

            if not is_recoverable:
                raise

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            corrupt_backup = f"{file_name}.corrupt_{timestamp}"

            try:
                os.replace(file_name, corrupt_backup)
                logging.warning(
                    "Corrupted H5 file detected at %s. Backed up to %s and recreated a clean store.",
                    file_name,
                    corrupt_backup,
                )
            except OSError:
                logging.warning(
                    "Corrupted H5 file detected at %s but backup rename failed. Recreating store in place.",
                    file_name,
                )

            return h5py.File(file_name, "a")


    def close(self):
        self.f.close()
                    
            
    def store_batch_embeddings(self, batch_embeddings: NDArray, batch_start_last_data: NDArray):
            
        group = self.f.require_group(self.dataset_name)
            
        embeddings = group["embeddings"]
            
        if isinstance(embeddings, Dataset):
            
            old_size = embeddings.shape[0]
            new_size = old_size + batch_embeddings.shape[0]
                
            embeddings.resize(new_size, axis=0)
                
            embeddings[old_size:new_size] = batch_embeddings.astype("float32")
            
            
        start_last_data = group["start_last_data"]
        
        if isinstance(start_last_data, Dataset):
            
            old_size = start_last_data.shape[0]
            new_size = old_size + batch_start_last_data.shape[0]
            
            start_last_data.resize(new_size, axis=0)
            
            start_last_data[old_size:new_size] = batch_start_last_data.astype("int")
            
    def get_one_embedding(self, frame_number: int) -> tuple[NDArray, NDArray]:

        group = self.f.require_group(self.dataset_name)
        embeddings = group["embeddings"]
        start_last_data = group["start_last_data"]
            
        if isinstance(embeddings, Dataset) and isinstance(start_last_data, Dataset):
            size = embeddings.shape[0]  
            size_start_last = start_last_data.shape[0]  
                
            if frame_number < size and frame_number < size_start_last:
                return (embeddings[frame_number], start_last_data[frame_number])
    
        raise Exception("An Error occured with getting a single embedding.")
    
    def is_data_present(self) -> bool:
        group = self.f.require_group(self.dataset_name)
        embeddings = group["embeddings"]
        
        if isinstance(embeddings, Dataset):
            size = embeddings.shape[0]
            return size > 0
        
        raise Exception("There was an error while checking if the dataset had some data in it or not.")
    
    def get_all_embeddings(self):
        
        group = self.f.require_group(self.dataset_name)
            
        embeddings = group["embeddings"]
        start_last_data = group["start_last_data"]

        if isinstance(embeddings, Dataset) and isinstance(start_last_data, Dataset):
            
            return embeddings[:], start_last_data[:]
            
        else:
                
            raise Exception("An error occured while getting all embeddings")
    
    
    def generate_batch_embeddings(self, batch_size: int) -> Generator[tuple[NDArray, NDArray], None, None]:

        group = self.f.require_group(self.dataset_name)
        embeddings = group["embeddings"] 
        start_last_data = group["start_last_data"]
            
        if isinstance(embeddings, Dataset) and isinstance(start_last_data, Dataset):
            
            n_rows = embeddings.shape[0]
                
            for i in range(0, n_rows, batch_size):
                    
                yield embeddings[i:i+batch_size], start_last_data[i:i+batch_size]
            
        else:
                
            raise Exception("An error occured when trying to yield embeddings")
            

if __name__ == "__main__":
    
    mystore = Embedding_Store(
        file_name="embeddings",
        dataset_name="people",
        embedding_dimension=512,
        chunking_size=256
    )

    batch = np.random.randn(32, 512).astype("float32")
    start_lasts = np.random.randint(0, 50, size=(32, 2))

    mystore.store_batch_embeddings(batch, start_lasts)

    # embeddings = mystore.get_all_embeddings()

    # print(embeddings)

    # print(np.array(embeddings))


    for embedding_batch, start_last_data in mystore.generate_batch_embeddings(16):
        
        print(embedding_batch.shape, start_last_data.shape)
        
    mystore.close()