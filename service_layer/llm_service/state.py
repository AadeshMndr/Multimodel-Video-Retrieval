from typing import TypedDict, Literal
from pydantic import BaseModel, Field


#################################### Analyzer ####################################

All_Path = Literal["clip", "xclip", "yolo", "audio", "ocr"]

class Analyzer_State(TypedDict):
    
    user_prompt: str
    
    logical_path: All_Path
    
    
def get_analyzer_state(user_text: str):
    
    return Analyzer_State( # type: ignore
        user_prompt=user_text,
    )

class Analyzer_Output_Format(BaseModel):
    
    logical_path: All_Path = Field(description="The name of the path to take")


###################################### AUDIO ####################################

class Audio_State(TypedDict):

    user_prompt: str

    refined_query: str


def get_audio_state(user_prompt: str) -> Audio_State:

    return Audio_State(  # type: ignore
        user_prompt=user_prompt
    )


class Audio_OutputFormat(BaseModel):

    refined_query: str = Field(description="A concise query focused on spoken content semantics")


###################################### OCR ####################################

class OCR_State(TypedDict):

    user_prompt: str

    regex_keywords: list[str]

    semantic_query: str


def get_ocr_state(user_prompt: str) -> OCR_State:

    return OCR_State(  # type: ignore
        user_prompt=user_prompt
    )


class OCR_OutputFormat(BaseModel):

    regex_keywords: list[str] = Field(
        description="Keywords or numbers likely to appear as on-screen text for regex search"
    )
    semantic_query: str = Field(
        description="Short literal phrase suitable for semantic OCR search"
    )

###################################### CLIP ####################################3



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
