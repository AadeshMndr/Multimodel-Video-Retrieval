from langgraph.graph import StateGraph, START, END
from service_layer.clip_service.state import State
from service_layer.clip_service.node_functions import generate_text_embeddings, generate_video_embeddings, filter_only_matched_frames, is_reassessment_required, reassess_scores

graph = StateGraph(State)

graph.add_node("embed_texts", generate_text_embeddings)
graph.add_node("embed_frames", generate_video_embeddings)
graph.add_node("match", filter_only_matched_frames)
graph.add_node("re-assess", reassess_scores)

graph.add_edge(START, "embed_texts")
graph.add_edge(START, "embed_frames")
graph.add_edge("embed_texts", "match")
graph.add_edge("embed_frames", "match")
graph.add_conditional_edges("match", is_reassessment_required)
graph.add_conditional_edges("re-assess", is_reassessment_required)

workflow = graph.compile()


