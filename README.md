# Multimodel Video Retrieval

A multimodal video retrieval backend built with FastAPI, PyTorch, and FAISS. Upload a video, ask a natural language query, and the system finds matching video segments using CLIP, xCLIP, YOLO, audio transcription, and OCR.


## Project Overview

This repository contains the backend API for a multimodal video retrieval system. It can analyze uploaded videos with semantic visual search, object detection, speech search, and OCR-based text retrieval. The backend exposes REST endpoints for uploading videos, submitting search prompts, downloading output clips, and fetching uploaded source videos.

## Features

- Upload videos and store them locally in `upload/`
- Query uploaded videos with natural language prompts
- Retrieve matching segments from:
  - CLIP / xCLIP visual embeddings
  - YOLO object detection
  - Audio speech transcription
  - OCR text extraction
- Generate and download clipped video outputs
- Optional on-upload indexing for audio and OCR
- Configurable model, device, and search thresholds

## Tech Stack

- Python 3.x
- FastAPI
- Uvicorn
- PyTorch
- FAISS
- Hugging Face Transformers
- SentenceTransformers
- EasyoCR
- Whisper / Faster Whisper
- OpenAI / GROQ LLM support via `groq/compound`
- CLIP / xCLIP
- YOLOv8-based object detection
- ffmpeg for audio/video muxing

## Requirements

- Python 3.10+ (recommended)
- ffmpeg installed on the system path
- GPU support optional: CUDA or Apple MPS
- `requirements.txt` dependencies installed

## Setup

1. Create a Python virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Configure environment variables:

- Copy or create a `.env` file in the repository root
- At minimum, add a valid `GROQ_API_KEY` if you use the routing/LLM features

Example `.env`:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
PORT=5050
DEVICE=cpu
```

## Environment Variables

The backend loads configuration from `.env` and system environment variables using `pydantic-settings`.

Important entries:

- `GROQ_API_KEY` — required for LLM-assisted query routing and prompt handling
- `PORT` — backend port (default: `5050`)
- `DEVICE` — override compute device: `cpu`, `cuda`, or `mps`
- `CALCULATE_EMBEDDINGS_ON_UPLOAD` — whether to build CLIP/xCLIP embeddings during upload
- `CALCULATE_AUDIO_INDEX_ON_UPLOAD` — whether to create audio indexes when uploading
- `CALCULATE_OCR_INDEX_ON_UPLOAD` — whether to create OCR indexes when uploading

# Run the backend server

```bash
python main.py
```


# Upload the file:

```bash
curl -X POST "http://127.0.0.1:5050/upload" -F "file=@people.mp4"
```
```bash
curl -X POST "http://127.0.0.1:5050/upload" -F "file=@C:\Users\Legion\Desktop\Video clip experiments\Spiderman.mp4"
```


# Give Prompt


```bash
curl -X POST "http://127.0.0.1:5050/prompt" \
-H 'Content-Type: application/json' \
-d '{ "prompt": "Find me clips with exactly two people", "filename": "people.mp4" }' \
--no-buffer
```
```bash
curl -N -X POST "http://localhost:5050/prompt" -H "Content-Type: application/json" -d "{\"prompt\":\"Find where the speaker tells differences between backpropagation and brain learning\",\"filename\":\"yt_video.mp4\"}"
```
```bash
curl -N -X POST "http://localhost:5050/prompt" -H "Content-Type: application/json" -d "{\"prompt\":\"Man in orange shirt\",\"filename\":\"Spiderman.mp4\"}"
```


# Download Video Clip

```bash
curl -o myclip.mp4 "http://127.0.0.1:5050/download/people.mp4"
```
