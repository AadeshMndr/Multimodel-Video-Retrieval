from langgraph.graph import StateGraph, START, END
from service_layer.yolo_service.state import State 
from service_layer.yolo_service.node_functions import decision_node, filter_only_matched_frames_that_match_all, filter_only_matched_frames_that_match_any, filter_only_matched_frames_in_canonical_form_POS, filter_only_matched_frames_in_canonical_form_SOP, is_reassessment_required, reassess_scores

graph = StateGraph(State)

# graph.add_node("match_any", filter_only_matched_frames_that_match_any)
# graph.add_node("match_all", filter_only_matched_frames_that_match_all)
graph.add_node("match_POS", filter_only_matched_frames_in_canonical_form_POS)
graph.add_node("match_SOP", filter_only_matched_frames_in_canonical_form_SOP)
graph.add_node("re-assess", reassess_scores)



graph.add_conditional_edges(START, decision_node)
# graph.add_edge("match_any", END)
# graph.add_edge("match_all", END)
graph.add_conditional_edges("match_POS", is_reassessment_required)
graph.add_conditional_edges("match_SOP", is_reassessment_required)

graph.add_conditional_edges("re-assess", decision_node)

workflow = graph.compile()