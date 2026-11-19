"""Extractor selection - the EXTRACTOR switch.

EXTRACTOR=pypdf  (default) -> extract_pdf.py, the pinned pypdf 6.15.0 path.
EXTRACTOR=tika             -> the parser helper at TIKA_URL: RocketRide's
                              own parser jars and configuration behind a
                              small HTTP server (langgraph/parser), so
                              both systems parse a PDF to the same text.

Pure computation layer: stdlib HTTP only, no fastapi/langgraph imports.
The mode is read once at import; changing it requires a service restart,
which keeps any single run single-mode by construction.
"""

import os
import urllib.request

EXTRACTOR = os.environ.get("EXTRACTOR", "pypdf").strip().lower()
TIKA_URL = os.environ.get("TIKA_URL", "http://tika:9998")


def extract_tika(source_path: str) -> str:
    """PDF -> text via the Tika server sidecar (PUT /tika, text/plain)."""
    with open(source_path, "rb") as fh:
        data = fh.read()
    req = urllib.request.Request(
        f"{TIKA_URL}/tika",
        data=data,
        method="PUT",
        headers={
            "Content-Type": "application/pdf",
            "Accept": "text/plain; charset=UTF-8",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    # The text is returned exactly as the parser gave it, leading and trailing
    # blank lines included: the chunk texts must match RocketRide's to the byte.
    return text


def extract(source_path: str) -> str:
    if EXTRACTOR == "tika":
        return extract_tika(source_path)
    from workload.document.extract_pdf import extract_pdf

    return extract_pdf(source_path)


def extractor_info() -> dict:
    info = {"mode": EXTRACTOR}
    if EXTRACTOR == "tika":
        info["tika_url"] = TIKA_URL
        info["parser"] = "RocketRide's parser jars and configuration (langgraph/parser)"
    else:
        from workload.document.extract_pdf import pypdf_version

        info["pypdf"] = pypdf_version()
    return info
