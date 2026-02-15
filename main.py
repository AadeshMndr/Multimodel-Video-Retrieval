from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import logging
from config import settings
import shutil
import os
from router.main_graph import main_workflow
from router.main_state import get_main_state, Main_State


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

app = FastAPI()

@app.get("/")
async def introduce():
    
    return "Upload video at /upload and Write Prompt at /prompt"






@app.post("/prompt")
async def find_in_video(prompt: str = Body(min_length=1), filename: str = Body(min_length=1)):

    
    if prompt.strip() == "":
        return HTTPException(status_code=400, detail=f"The prompt cannot be empty")
    
    
    if not os.path.exists(f"upload/{filename}"):
        return HTTPException(status_code=400, detail=f"The file {filename} could not be found, please upload it first.")
    
    os.makedirs("outputs", exist_ok=True)
    
    initial_state = get_main_state(
        video_path=f"upload/{filename}",
        user_text=prompt,
        output_path=f"outputs/output_{filename}"
    )
    
    final_state: Main_State = main_workflow.invoke(initial_state) # type: ignore
    
    return JSONResponse(
        content={
            "message": f"The output is stored at output_{final_state["output_path"]}",
            "matched_frames" : len(final_state["video_state"]),
            "id": "1",
        }
    )
    
    


@app.post("/upload")
async def upload_video(file: UploadFile = File()):
    
    os.makedirs("upload", exist_ok=True)
    
    with open(f"upload/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return { "message": f"{file.filename} Upload Complete!" }


@app.get("/download/{filename}")
async def download_clip(filename: str):
    
    if  not os.path.exists(f"outputs/output_{filename}"):
        return HTTPException(status_code=400, detail=f"The file {filename} could not be found, please process the video by giving a prompt first")

    return FileResponse(path=f"outputs/output_{filename}", filename=f"output-for-{filename}", media_type="application/octet-stream")

if __name__ == "__main__":
    logging.info(f"Starting the web server at port {settings.PORT}")
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=settings.PORT)