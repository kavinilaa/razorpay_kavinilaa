import secrets

from fastapi import Header, HTTPException, status

from .config import settings

# Deliberately the SAME message and status code whether the header was
# missing entirely or present but wrong, so a caller cannot distinguish
# "no key sent" from "wrong key sent" from the response body alone.
UNAUTHORIZED_DETAIL = "Unauthorized"


def require_api_key(
    api_key_header: str | None = Header(default=None, alias=settings.api_key_header_name),
) -> None:
    """FastAPI dependency - raises 401 unless `api_key_header` matches
    `settings.api_key`. Returns None (no value) on success; callers don't
    need the key's value, just the fact that it was valid.

    Uses `secrets.compare_digest` (constant-time for equal-length inputs)
    rather than `==`, and normalizes a missing header to an empty string
    before comparing, so a missing key and a wrong key take the exact same
    code path - not just the same status code and body, but the same
    comparison work, to avoid a timing oracle in addition to a content one.
    This is a best-effort mitigation, not a formal guarantee, given HTTP's
    many other sources of timing noise (network, ASGI stack, GC pauses).
    """
    supplied = api_key_header or ""
    if not secrets.compare_digest(supplied, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UNAUTHORIZED_DETAIL)
