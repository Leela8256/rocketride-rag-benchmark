"""search-pdf-rag-v1: Pipeline protocol adapter over the compiled graph.

A request carries no file. The question and the number of results come in the request's options:
    {"question": "...", "k": 10}
"""

from typing import Any, Dict

from pipelines.search_pdf_rag.graph import build_search_graph
from service.errors import ServiceError
from service.pipeline import RequestContext
from workload.document.embed import load_model
from workload.document.store_pg import load_store

DEFAULT_K = 10


class SearchPdfRagPipeline:
    name = "search-pdf-rag-v1"

    def __init__(self) -> None:
        # Compiled ONCE, at construction, never per request.
        self.graph = build_search_graph()

    async def warmup(self) -> None:
        """Load the model, open the store and run one question through the FULL graph.

        The table may still be empty at start, so an empty result is fine; a failing query is not.
        """
        load_model()
        load_store()
        final = await self.graph.ainvoke({"question": "warm-up question", "k": DEFAULT_K, "timings_ns": {}})
        if not isinstance((final.get("result") or {}).get("documents"), list):
            raise RuntimeError(f"{self.name} warmup returned no result list")

    async def prepare_input(self, context: RequestContext) -> Dict[str, Any]:
        question = context.options.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ServiceError("empty_input", f"pipeline {self.name!r} requires a question", context.request_id)
        k = context.options.get("k", DEFAULT_K)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 1000:
            raise ServiceError("invalid_options", "k must be an integer from 1 to 1000", context.request_id)
        return {"question": question, "k": k, "timings_ns": {}}

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self.graph.ainvoke(state)

    async def extract_output(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        return final_state["result"]


def build_search_pdf_rag_pipeline() -> SearchPdfRagPipeline:
    return SearchPdfRagPipeline()
