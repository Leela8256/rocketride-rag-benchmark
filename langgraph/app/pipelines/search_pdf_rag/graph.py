"""The compiled LangGraph for search-pdf-rag-v1.

Node names mirror the RocketRide search pipe's COMPUTE components one-for-one:
    embed     ~ embedding_transformer_1
    retrieve  ~ postgres_1

No checkpointer: this is a stateless request/response pipeline.
"""

from langgraph.graph import END, START, StateGraph

from pipelines.search_pdf_rag.nodes import embed_node, retrieve_node
from pipelines.search_pdf_rag.state import SearchState


def build_search_graph():
    b = StateGraph(SearchState)
    b.add_node("embed", embed_node)
    b.add_node("retrieve", retrieve_node)
    b.add_edge(START, "embed")
    b.add_edge("embed", "retrieve")
    b.add_edge("retrieve", END)
    return b.compile()
