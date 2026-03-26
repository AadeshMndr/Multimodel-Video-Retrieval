from langgraph.graph import StateGraph, START, END
from service_layer.llm_service.state import Modified_Prompts_State, YOLO_State, Analyzer_State, Audio_State, OCR_State
from service_layer.llm_service.node_functions import generate_similar_prompts, generate_synonyms, analyze_the_prompt, refine_audio_prompt, refine_ocr_prompt



###################### Analyzer ################################



analyzer_graph = StateGraph(Analyzer_State)

analyzer_graph.add_node("analyze", analyze_the_prompt)

analyzer_graph.add_edge(START, "analyze")
analyzer_graph.add_edge("analyze", END)

analyzer_workflow = analyzer_graph.compile()


###################### AUDIO ################################

audio_graph = StateGraph(Audio_State)

audio_graph.add_node("refine_audio_prompt", refine_audio_prompt)

audio_graph.add_edge(START, "refine_audio_prompt")
audio_graph.add_edge("refine_audio_prompt", END)

audio_workflow = audio_graph.compile()


###################### OCR ################################

ocr_graph = StateGraph(OCR_State)

ocr_graph.add_node("refine_ocr_prompt", refine_ocr_prompt)

ocr_graph.add_edge(START, "refine_ocr_prompt")
ocr_graph.add_edge("refine_ocr_prompt", END)

ocr_workflow = ocr_graph.compile()


###################### CLIP ################################

prompt_variation_graph = StateGraph(Modified_Prompts_State)

prompt_variation_graph.add_node("generate_prompt_variations", generate_similar_prompts)

prompt_variation_graph.add_edge(START, "generate_prompt_variations")
prompt_variation_graph.add_edge("generate_prompt_variations", END)

prompt_variation_workflow = prompt_variation_graph.compile()


###################### YOLO ################################

yolo_graph = StateGraph(YOLO_State)

yolo_graph.add_node("generate_synonyms_and_boolean_logic", generate_synonyms)

yolo_graph.add_edge(START, "generate_synonyms_and_boolean_logic")
yolo_graph.add_edge("generate_synonyms_and_boolean_logic", END)

yolo_workflow = yolo_graph.compile()
