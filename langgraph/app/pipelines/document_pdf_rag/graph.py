"""The compiled LangGraph for document-pdf-rag-v1: a RAG ingestion pipeline.

Node names mirror the RocketRide pipe's COMPUTE components one-for-one:
    extract  ~ parse_1
    chunk    ~ preprocessor_1
    embed    ~ embedding_1
    store    ~ postgres_1

The store is the last node, as in the RocketRide ingestion pipe, where the store is the terminal component and
there is no response component. The response carries only the number of rows written.

No checkpointer: this is a stateless request/response pipeline.
"""

from langgraph.graph import END, START, StateGraph

from pipelines.document_pdf_rag.nodes import chunk_node, embed_node, extract_node, store_node
from pipelines.document_pdf_rag.state import PdfRagState


def build_pdf_rag_graph():
    b = StateGraph(PdfRagState)
    b.add_node("extract", extract_node)
    b.add_node("chunk", chunk_node)
    b.add_node("embed", embed_node)
    b.add_node("store", store_node)
    b.add_edge(START, "extract")
    b.add_edge("extract", "chunk")
    b.add_edge("chunk", "embed")
    b.add_edge("embed", "store")
    b.add_edge("store", END)
    return b.compile()
