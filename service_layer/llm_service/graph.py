from langgraph.graph import StateGraph, START, END
from service_layer.llm_service.state import State
from service_layer.llm_service.node_functions import generate_similar_prompts

graph = StateGraph(State)

graph.add_node("generate_prompt_variations", generate_similar_prompts)

graph.add_edge(START, "generate_prompt_variations")
graph.add_edge("generate_prompt_variations", END)

workflow = graph.compile()