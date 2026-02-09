from service_layer.llm_service.state import State, OutputFormat
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from infrastructure.llm import llm
from config import settings
import logging

parser = PydanticOutputParser(pydantic_object=OutputFormat)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

def generate_similar_prompts(state: State):
    
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
            "format_instructions": parser.get_format_instructions()
        }
    )    
    
    chain = template | llm | parser
    
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
    