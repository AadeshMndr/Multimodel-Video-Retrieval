from typing import TypedDict, Literal
from pydantic import BaseModel, Field

class Modified_Prompts_State(TypedDict):
    
    user_prompt: str 
    
    modified_prompts: list[str]
    

def get_modified_prompt_state(user_text: str) -> Modified_Prompts_State:
    
    return Modified_Prompts_State(
        user_prompt=user_text,
        modified_prompts=[]
    )

class OutputFormat(BaseModel):
    
    modified_prompts: list[str] = Field(description="list of modified prompts")

    
    
    


################### YOLO ######################

class YOLO_State(TypedDict):
    
    user_prompt: str 
    
    object_details: dict[str, tuple[float, float]]

    object_groups: list[list[str]]
    
    canonical_form: Literal["SOP", "POS"]
     

def get_yolo_state(user_prompt: str):
    
    return YOLO_State(   # type: ignore
        user_prompt=user_prompt
    )
    
class YOLO_OutputFormat(BaseModel):
    
    object_details: dict[str, tuple[float, float]] = Field(description="A dictionary that has key: (min_count, max_count)")

    object_groups: list[list[str]] = Field(description="The object names grouped according to their boolean logic")
    
    canonical_form: Literal["SOP", "POS"] = Field(description="The name of the canonical form the boolean logic is in, i.e POS or SOP")
    

    

    
    
    