from pydantic import BaseModel
from typing import Literal

class StreamOutput(BaseModel):
    status: Literal["initiating", "processing", "creating_video", "complete"]
    timestamps: list[tuple[int, int]]
    