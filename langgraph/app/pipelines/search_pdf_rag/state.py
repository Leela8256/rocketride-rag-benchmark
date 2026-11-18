"""Graph state for search-pdf-rag-v1: one question in, the nearest stored chunks out."""

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class SearchState(TypedDict, total=False):
    question: str
    k: int
    vector: List[float]
    timings_ns: Dict[str, int]
    result: Optional[Dict[str, Any]]
