from langgraph.graph import END, START, StateGraph

from service_layer.xclip_service.node_functions import find_temporal_matches, is_reassessment_required, reassess_matches
from service_layer.xclip_service.state import State


graph = StateGraph(State)

graph.add_node("find_temporal_matches", find_temporal_matches)
graph.add_node("re-assess", reassess_matches)

graph.add_edge(START, "find_temporal_matches")
graph.add_conditional_edges("find_temporal_matches", is_reassessment_required)
graph.add_conditional_edges("re-assess", is_reassessment_required)

workflow = graph.compile()
