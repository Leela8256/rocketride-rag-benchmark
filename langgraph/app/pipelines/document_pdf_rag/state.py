"""Graph state for document-pdf-rag-v1: the PDF state plus the name the rows are stored under."""

from pipelines.document_pdf.state import PdfState


class PdfRagState(PdfState, total=False):
    source_name: str
