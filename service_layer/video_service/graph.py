from langgraph.graph import StateGraph, START, END
from service_layer.video_service.state import State
from service_layer.video_service.node_functions import sample_frames, remove_similar_frames, expand_frame_range, batch_frames, clip_video


pre_graph = StateGraph(State)

pre_graph.add_node("sample", sample_frames)
pre_graph.add_node("remove_similar", remove_similar_frames)
pre_graph.add_node("batch", batch_frames)

pre_graph.add_edge(START, "sample")
pre_graph.add_edge("sample", "remove_similar")
pre_graph.add_edge("remove_similar", "batch")
pre_graph.add_edge("batch", END)

pre_workflow = pre_graph.compile()


post_graph = StateGraph(State)

post_graph.add_node("expand", expand_frame_range)
post_graph.add_node("clip", clip_video)

post_graph.add_edge(START, "expand")
post_graph.add_edge("expand", "clip")
post_graph.add_edge("clip", END)

post_workflow = post_graph.compile()