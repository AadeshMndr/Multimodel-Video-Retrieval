from langgraph.graph import END, START, StateGraph

from service_layer.audio_service.node_functions import ensure_index, query_timestamps
from service_layer.audio_service.state import State

graph = StateGraph(State)

graph.add_node("ensure_index", ensure_index)
graph.add_node("query_timestamps", query_timestamps)

graph.add_edge(START, "ensure_index")
graph.add_edge("ensure_index", "query_timestamps")
graph.add_edge("query_timestamps", END)

workflow = graph.compile()
