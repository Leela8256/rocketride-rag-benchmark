"""Nodes of document-pdf-rag-v1.

extract, chunk and embed are the document-pdf-v1 nodes themselves, imported, not copied: the same parser, the
same splitter and the same shared model. Only the last node is new. It writes the document's chunks and vectors
to the vector store and returns the count.

Plain sync functions, like every other node: LangGraph runs them on the default thread pool. Nothing here adds a
lock, a limit, a queue or a batch.
"""

import time
from typing import Any, Dict

from pipelines.document_pdf.nodes import _timed, chunk_node, embed_node, extract_node  # noqa: F401
from pipelines.document_pdf_rag.state import PdfRagState
from workload.document.store_pg import store_chunks


def store_node(state: PdfRagState) -> Dict[str, Any]:
    t0 = time.perf_counter_ns()
    stored = store_chunks(state["source_name"], state["chunks"], state["vectors"])
    result = {"stored": stored, "chunks": len(state["chunks"])}
    return {"result": result, "timings_ns": _timed(state, "store", t0)}
