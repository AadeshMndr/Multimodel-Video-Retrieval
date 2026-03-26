from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import uvicorn
import logging
from config import settings
import shutil
import os
from router.main_graph import upload_workflow
from router.main_state import get_main_state, Main_State
from api.processing_logic import process_the_video
from infrastructure.audio_video_indexer import Audio_Video_Indexer
from infrastructure.ocr_indexer import OCR_Indexer
from infrastructure.mac_gpu_utils import setup_mac_gpu_environment, print_device_info


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

# Setup Mac GPU environment if running on Apple Silicon
setup_mac_gpu_environment()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def introduce():
    
    return "Upload video at /upload and Write Prompt at /prompt"




@app.post("/prompt")
async def find_in_video(prompt: str = Body(min_length=1), filename: str = Body(min_length=1), generate_output_video: bool = Body(default=True)):

    
    if prompt.strip() == "":
        return HTTPException(status_code=400, detail=f"The prompt cannot be empty")
    
    
    video_path = f"upload/{filename}"
    
    if not os.path.exists(video_path):
        return HTTPException(status_code=400, detail=f"The file {filename} could not be found, please upload it first.")
    
    os.makedirs("outputs", exist_ok=True)
    
    output_path = f"outputs/output_{filename}"
    
    # initial_state = get_main_state(
    #     video_path=video_path,
    #     user_text=prompt,
    #     output_path=output_path
    # )
    
    # final_state: Main_State = main_workflow.invoke(initial_state) # type: ignore
    
    # return JSONResponse(
    #     content={
    #         "message": f"The output is stored at output_{final_state["output_path"]}",
    #         "matched_frames" : len(final_state["video_state"]),
    #         "id": "1",
    #     }
    # )
    
    return StreamingResponse(
        process_the_video(
            video_path=video_path,
            user_text=prompt,
            output_path=output_path,
            generate_output_video=generate_output_video,
        ),
        media_type="text/plain"
    )
    


@app.post("/upload")
async def upload_video(file: UploadFile = File()):
    
    os.makedirs("upload", exist_ok=True)
    
    upload_path = f"upload/{file.filename}"
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if settings.CALCULATE_EMBEDDINGS_ON_UPLOAD or settings.CALCULATE_XCLIP_EMBEDDINGS_ON_UPLOAD:
        
        logging.info("Generating and storing the embeddings during upload...")

        main_state = get_main_state(
            video_path=upload_path,
            user_text="",
            output_path=""
        )

        main_state["logical_path_choosen"] = "clip"
        
        upload_workflow.invoke(main_state)

    if settings.CALCULATE_AUDIO_INDEX_ON_UPLOAD:
        logging.info("Transcribing and indexing audio during upload...")

        video_name = os.path.splitext(os.path.basename(upload_path))[0]
        index_dir = settings.AUDIO_INDEX_DIR
        index_path = os.path.join(index_dir, f"{video_name}.faiss")
        meta_path = os.path.join(index_dir, f"{video_name}.pkl")

        Audio_Video_Indexer().ensure_index(
            video_path=upload_path,
            index_path=index_path,
            meta_path=meta_path,
        )

    if settings.CALCULATE_OCR_INDEX_ON_UPLOAD:
        logging.info("Indexing OCR text during upload...")

        video_name = os.path.splitext(os.path.basename(upload_path))[0]
        index_dir = settings.OCR_INDEX_DIR
        index_path = os.path.join(index_dir, f"{video_name}.faiss")
        meta_path = os.path.join(index_dir, f"{video_name}.pkl")
        transcript_dir = settings.OCR_TRANSCRIPT_DIR
        transcript_path = os.path.join(transcript_dir, f"{video_name}.txt")

        OCR_Indexer().ensure_index(
            video_path=upload_path,
            index_path=index_path,
            meta_path=meta_path,
            transcript_path=transcript_path,
        )
        
    return { "message": f"{file.filename} Upload Complete!" }


@app.get("/download/{filename}")
async def download_clip(filename: str):
    
    if  not os.path.exists(f"outputs/output_{filename}"):
        return HTTPException(status_code=400, detail=f"The file {filename} could not be found, please process the video by giving a prompt first")

    return FileResponse(path=f"outputs/output_{filename}", filename=f"output-for-{filename}", media_type="application/octet-stream")

if __name__ == "__main__":
    # Print device information at startup
    print_device_info()
    logging.info(f"Using device: {settings.DEVICE}")
    logging.info(f"Starting the web server at port {settings.PORT}")
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=settings.PORT)
