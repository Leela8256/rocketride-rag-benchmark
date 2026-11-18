"""document-pdf-rag-v1: Pipeline protocol adapter over the compiled graph."""

from typing import Any, Dict

from pipelines.document_pdf.adapter import PDF_MEDIA_TYPE, WARMUP_PDF
from pipelines.document_pdf_rag.graph import build_pdf_rag_graph
from service.errors import ServiceError
from service.pipeline import RequestContext
from workload.document.embed import load_model
from workload.document.store_pg import load_store


class DocumentPdfRagPipeline:
    name = "document-pdf-rag-v1"

    def __init__(self) -> None:
        # Compiled ONCE, at construction, never per request.
        self.graph = build_pdf_rag_graph()

    async def warmup(self) -> None:
        """Load the model, open the store and run a real PDF through the FULL graph, the write included.

        Anything that fails here aborts startup: a pipeline that cannot store its own fixture must never be
        advertised as ready. The fixture's rows stay in the table under the name warmup.pdf.
        """
        load_model()
        load_store()
        state = {"source_path": str(WARMUP_PDF), "source_name": WARMUP_PDF.name, "timings_ns": {}}
        final = await self.graph.ainvoke(state)
        result = final.get("result")
        if not result or not result.get("stored") or result["stored"] != result["chunks"]:
            raise RuntimeError(f"{self.name} warmup stored no rows from {WARMUP_PDF}")

    async def prepare_input(self, context: RequestContext) -> Dict[str, Any]:
        if context.source_path is None:
            raise ServiceError("empty_input", f"pipeline {self.name!r} requires a file", context.request_id)
        if context.media_type != PDF_MEDIA_TYPE:
            raise ServiceError(
                "unsupported_media_type",
                f"pipeline {self.name!r} accepts only {PDF_MEDIA_TYPE}, got {context.media_type!r}",
                context.request_id,
            )
        return {
            "source_path": context.source_path,
            "source_name": context.filename or context.request_id,
            "timings_ns": {},
        }

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self.graph.ainvoke(state)

    async def extract_output(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        return final_state["result"]


def build_document_pdf_rag_pipeline() -> DocumentPdfRagPipeline:
    return DocumentPdfRagPipeline()
