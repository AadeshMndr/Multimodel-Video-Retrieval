from langgraph.graph import StateGraph, START, END
from router.main_state import Main_State
from router.main_logic import prompt_analyzer, postprocess, preprocess, decide_route, parallel_post_process
from router.logic_routes.clip_logic import clip_logic, generate_and_store_embeddings
from router.logic_routes.xclip_logic import xclip_logic, generate_and_store_xclip_embeddings
from router.logic_routes.yolo_logic import yolo_logic
from router.logic_routes.audio_logic import audio_logic
from router.logic_routes.ocr_logic import ocr_logic

main_graph = StateGraph(Main_State)

main_graph.add_node("prompt analyzer", prompt_analyzer)
main_graph.add_node("preprocess", preprocess)
# main_graph.add_node("postprocess", postprocess)
main_graph.add_node("postprocess", parallel_post_process)

main_graph.add_node("clip", clip_logic)
main_graph.add_node("xclip", xclip_logic)
main_graph.add_node("yolo", yolo_logic)
main_graph.add_node("audio", audio_logic)
main_graph.add_node("ocr", ocr_logic)

main_graph.add_edge(START, "prompt analyzer")
main_graph.add_edge("prompt analyzer", "preprocess")
main_graph.add_conditional_edges("preprocess", decide_route)
main_graph.add_edge("clip", "postprocess")
main_graph.add_edge("xclip", "postprocess")
main_graph.add_edge("yolo", "postprocess")
main_graph.add_edge("audio", "postprocess")
main_graph.add_edge("ocr", "postprocess")
main_graph.add_edge("postprocess", END)

main_workflow = main_graph.compile()



upload_graph = StateGraph(Main_State)

upload_graph.add_node("preprocess", preprocess)
upload_graph.add_node("generate_and_store_embeddings", generate_and_store_embeddings)
upload_graph.add_node("generate_and_store_xclip_embeddings", generate_and_store_xclip_embeddings)

upload_graph.add_edge(START, "preprocess")
upload_graph.add_edge("preprocess", "generate_and_store_embeddings")
upload_graph.add_edge("generate_and_store_embeddings", "generate_and_store_xclip_embeddings")
upload_graph.add_edge("generate_and_store_xclip_embeddings", END)

upload_workflow = upload_graph.compile()
