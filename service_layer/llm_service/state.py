from typing import TypedDict
from pydantic import BaseModel, Field

class State(TypedDict):
    
    user_prompt: str 
    
    modified_prompts: list[str]
    

def get_state(user_text: str) -> State:
    
    return State(
        user_prompt=user_text,
        modified_prompts=[]
    )

class OutputFormat(BaseModel):
    
    modified_prompts: list[str] = Field(description="list of modified prompts")

    

    
    
    