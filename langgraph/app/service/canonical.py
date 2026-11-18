"""Canonical JSON encoder - the single source of response bytes.

Every response body is encoded here, with fixed flags, so that the same
result always produces the same bytes and the SHA-256 the service returns
in X-Output-SHA256 can be checked by the client.
"""

import hashlib
import json

SEPARATORS = (",", ":")
ENSURE_ASCII = True
SORT_KEYS = False
ALLOW_NAN = False


class CanonicalEncodingError(Exception):
    """Raised when a value cannot be canonically encoded (NaN/Inf/non-serializable)."""


def canonical_encode(obj: object) -> bytes:
    """Encode obj to canonical UTF-8 JSON bytes. Never emits NaN/Inf."""
    try:
        text = json.dumps(
            obj,
            separators=SEPARATORS,
            ensure_ascii=ENSURE_ASCII,
            sort_keys=SORT_KEYS,
            allow_nan=ALLOW_NAN,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalEncodingError(str(exc)) from exc
    return text.encode("utf-8")


def canonical_sha256(data: bytes) -> str:
    """Hex sha256 of canonical bytes - what X-Output-SHA256 carries."""
    return hashlib.sha256(data).hexdigest()
