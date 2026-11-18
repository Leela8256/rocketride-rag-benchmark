"""Nodes of search-pdf-rag-v1, the search side of the RAG pipeline.

Two steps, mirroring the RocketRide search pipe (chat -> embedding_transformer -> postgres):
    embed     the question becomes a vector, with the same shared model the ingestion graph uses
    retrieve  the nearest stored chunks, through LangChain's PGVectorStore

Plain sync functions on the default thread pool. No lock, limit, cache or re-ranking is added.
"""

import time
from typing import Any, Dict

from pipelines.document_pdf.nodes import _timed
from pipelines.search_pdf_rag.state import SearchState
from workload.document.embed import embed_chunks
from workload.document.store_pg import search_chunks


def embed_node(state: SearchState) -> Dict[str, Any]:
    t0 = time.perf_counter_ns()
    vector = embed_chunks([state["question"]])[0]
    return {"vector": vector, "timings_ns": _timed(state, "embed", t0)}


def retrieve_node(state: SearchState) -> Dict[str, Any]:
    t0 = time.perf_counter_ns()
    documents = search_chunks(state["vector"], state["k"])
    return {"result": {"documents": documents}, "timings_ns": _timed(state, "retrieve", t0)}
