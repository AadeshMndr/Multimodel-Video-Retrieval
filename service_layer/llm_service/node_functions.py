from service_layer.llm_service.state import Modified_Prompts_State, OutputFormat, YOLO_State, YOLO_OutputFormat, Analyzer_Output_Format, Analyzer_State
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from infrastructure.llm import llm
from config import settings
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')



######################### analyzer #############################

analyzer_output_parser = PydanticOutputParser(pydantic_object=Analyzer_Output_Format)

def analyze_the_prompt(state: Analyzer_State):
    
    template = PromptTemplate(
        template="""
        This is the user's prompt: {user_prompt}
        
        The user has also provided a video, and now based on the prompt above, we need to find those
        clips that matches the description.
        
        I have the following tools:
        
        1) Semantic Video Searcher ("clip"): 
            This component internally uses clip to find the relevent clips.
            This component can understand the semantic meaning of each frame,
            And hence, this is useful for prompts where the user searches for scenes in the video
            using descriptive texts about the scene or things in the video.
            like: "a person riding a horse", "a person wearing an orange shirt", "a brown dog with long ears", "etc"
            
            If you decide to use this return "clip".
            
        2) Object Presence Detector ("yolo"):
            This component internally uses a YOLO model.
            This component can detect presence or absence of objects from each frame to find the relevent clips.
            And hence, this is useful for prompts where the user wants to search for a appearance  / presence of certain objects
            or even certain combination of objects within the video.
            like: "one person and one flower", "a dot or a cat", "animal", "horses", etc
        
            If you decide to use this return "yolo"
            
            {format_instructions}
            
            Just include the answer in the format specified above, don't include any extra fluff or explanations.
        """,
        input_variables=["user_prompt"],
        partial_variables={
            "format_instructions": analyzer_output_parser.get_format_instructions()
        }
    )
    
    chain = template | llm | analyzer_output_parser
    
    output = chain.invoke({
        "user_prompt": state["user_prompt"],
    })
    
    logging.info("=" * 60)
    logging.info(f"\nThe choosen path: {output.logical_path}\n")
    logging.info("=" * 60)
    
    return { "logical_path", output.logical_path }





########################### CLIP ###############################

modified_prompt_parser = PydanticOutputParser(pydantic_object=OutputFormat)

def generate_similar_prompts(state: Modified_Prompts_State):
    
    template = PromptTemplate(
        template="""
        This is the user's prompt: {user_prompt},
        
        Now, this prompt will be used to extract frames from a video that match it using CLIP.
        I want you to create a few more prompts that tries to say the same thing.
        If the user said: "Find me clips of a person walking with a blue shirt",
        Then you should give modified prompts such as: "Person in a blue shirt", "Person walking with a blue shirt", etc
        I just want you to create different versions of the same prompt by removing the things such as: "Find me clips...", "Clips of..." and such things.
        
        At max generate {max_num_of_modified_prompts} modified prompts.
        You don't always need to generate {max_num_of_modified_prompts} modified prompts, sometimes, if there is not many ways to say
        the same thing then, just generate how much you can. Give more priority to the semantic meaning of the modified prompts than 
        the number of modified prompts.
        
        {format_instructions}
        
        Just include the answer in the format specified above, don't include any extra fluff or explanations.
        """,
        input_variables=["user_prompt", "max_num_of_modified_prompts"],
        partial_variables={
            "format_instructions": modified_prompt_parser.get_format_instructions()
        }
    )    
    
    chain = template | llm | modified_prompt_parser
    
    output: OutputFormat = chain.invoke({ 
        "user_prompt": state["user_prompt"],
        "max_num_of_modified_prompts": settings.MAX_NUMBER_OF_MODIFIED_PROMPTS
        
    })
   
    logging.info("\n") 
    logging.info("=" * 60) 
    logging.info("Modified Prompts: ")
    
    for prompt in output.modified_prompts:
        logging.info(prompt)
        
    logging.info("=" * 60) 
    logging.info("\n") 
    
    return { "modified_prompts": output.modified_prompts }
   
   
   
   
   
#################### YOLO ##########################


yolo_parser = PydanticOutputParser(pydantic_object=YOLO_OutputFormat)   
   
   
def generate_synonyms(state: YOLO_State):
    
    template = PromptTemplate(
        template="""
        This is the user's prompt: {user_prompt},
        
        Now, this prompt will be used to extract frames from a video that has the object(s) described in it, using YOLO.
        I want you to give some synonyms of the object that is described in the prompt so that I can increase my chance of finding it using YOLO
       
        Don't keep things such as: "Find me clips...", "Clips of..." and such things from the prompt.
        
        At max generate {max_num_of_similar_words} synonyms for each object. Just generate the synonyms if the object name covers a wide variety of things and the synonym either narrows the range down or exactly describes the same word in another way.
        For example: 
        - If the word was "person" then you can generate "man, woman, boy, girl", etc.
        - If the word was "man" then you can generate "boy" but not "woman", "person" or "girl".
        
        After you decide which object names to keep, if the number of the object is specified in the prompt then include that information:
        
        Example:
            "User Prompt": "find me clips where there are 3 cats"
            
            "Your answer": 
                object_details = {{ 
                    "cat": (3.0, 3.0),
                    "feline": (3.0, 3.0),
                    "pussy cat": (3.0, 3.0),
                }}
                
            If the number of objects is not specified then just assume at least one.
                
            (This is just to explain what kind of answer I want, the actual format you will follow will be described later in this description)
        
        If the prompt requires a boolean expression of object presence then you must give that information as well:
        
        Example:
            "User Prompt": "find me clips of a cat and a man"
            
            "The boolean logic you thought of": ("cat" or "feline" or "pussy cat") and ("man" or "boy")
            
            "Your answer":
            
                object_groups = [ [ "cat", "feline", "pussy cat" ], [ "man", "boy" ] ]
                
                and canonical_form = "POS" (Product of Sum == AND of ORs)
                
        An Example where boolean logic is not required:
        
            "User Prompt": "find me clips of a woman"
            
            "Your answer":
            
                object_groups = [ [ "woman", "girl", "mother", "daughter" ] ]
                
                and canonical_form = "POS" (Here I assumed (("woman" or "girl" or "mother" or "daugther") and True and True))

        An Example for prompts where we are trying to find clips of things that are not present:
        
            "User Prompt": "find me clips with no people"
            
             "Your answer":
            
                object_groups = [ [ "woman", "girl", "mother", "daughter" ] ]
                
                and canonical_form = "SOP" (Here I assumed (("woman" and "girl" and "mother" and "daugther") or False or False))
                
                and the object_details dictionary should contain (0.0, 0.0) for each word
                
        Another example where the user wants clips where something is not present.
        
            "User Prompt": "one woman with no flowers"
            
            "Your answer":
            
                object_groups = [ [ "woman", "girl", "mother", "daughter" ], ["flower"], ["rose"], ["bouquet"] ]
                
                canonical_form = "POS" 
                
                aobject_details = {{ 
                    "woman": (1.0, 1.0),
                    "girl": (1.0, 1.0),
                    "mother": (1.0, 1.0),
                    "daugher": (1.0, 1.0),
                    "flower": (0.0, 0.0),
                    "rose": (0.0, 0.0),
                    "bouquet": (0.0, 0.0),
                }}

       
        Hence, at the end you answer should always contain:
            - object_details dictionary with the information about count of objects
            - object_groups list with the grouping information
            - canonical_form literal with the information about the boolean form

        MUST BE FOLLOWED: The object names present in `object_groups` should each have a corresponding entry in the `object_details`
        
        {format_instructions}
        
        Just include the answer in the format specified above, don't include any extra fluff or explanations.
        """,
        input_variables=["user_prompt", "max_num_of_similar_words"],
        partial_variables={
            "format_instructions": yolo_parser.get_format_instructions()
        }
    )    
    
    chain = template | llm | yolo_parser
    
    output: YOLO_OutputFormat = chain.invoke({ 
        "user_prompt": state["user_prompt"],
        "max_num_of_similar_words": settings.MAX_NUM_OF_SYNONYMS
    })
   
    logging.info("\n") 
    logging.info("=" * 60) 
    logging.info("Yolo input data: \n")
    
    logging.info(output.object_details)
    logging.info("\n")
    logging.info(f" Groups: {output.object_groups} \n")
    logging.info(f"Canonical form: {output.canonical_form}")
   
    logging.info("\n")
        
    logging.info("=" * 60) 
    logging.info("\n") 
    
    return { "object_details": output.object_details, "object_groups": output.object_groups, "canonical_form": output.canonical_form }