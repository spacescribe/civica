from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    rephrased_question: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str

from langgraph_pipeline.nodes.rephrase import rephrase_node
from langgraph_pipeline.nodes.retrieve import retrieve_node
from langgraph_pipeline.nodes.rerank import rerank_node
from langgraph_pipeline.nodes.answer import answer_node

graph_builder = StateGraph(GraphState)

graph_builder.add_node("rephrase", rephrase_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("answer", answer_node)

graph_builder.add_edge(START, "rephrase")
graph_builder.add_edge("rephrase", "retrieve")
graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "answer")
graph_builder.add_edge("answer", END)

graph = graph_builder.compile()