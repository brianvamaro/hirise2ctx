"""Shared download-commit guard (R66).

Four call sites in this repo independently implemented the same pattern — stream an HTTP
response into a `.tmp`/`.partial` sibling, then `Path.replace` it onto the destination —
and all four committed **unconditionally**. `HTTPResponse.read(amt)` does not raise on a
premature EOF, so a connection that dies mid-transfer yields a short file that is then
published into a permanent cache and thereafter trusted.

That is the same shape of defect as R23 (three byte-truncated `.shp` files whose missing
tail read as null geometry) and it is worse here for JP2: **GDAL does not raise on a
truncated JPEG2000 codestream — it silently zero-fills the missing region**, and Stage 2
converts that zero-fill straight into "no HiRISE coverage".

The atomic rename was never the problem; the *unconditional* rename was. This module holds
the one check that has to happen between the two. See DECISIONS 2026-08-06t.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def content_length_of(resp) -> int | None:
    """`Content-Length` as an int, or None when absent/unparseable.

    Absent is a real case, not a defect: a chunked or `Connection: close` response may omit
    it, and then a short read is genuinely indistinguishable from a complete one at the
    transport layer. That is exactly why callers should also pass a `validate` that inspects
    the bytes themselves.
    """
    try:
        raw = resp.headers.get("Content-Length")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def verify_download(
    tmp: str | Path,
    *,
    url: str,
    declared_length: int | None = None,
    min_bytes: int | None = None,
    validate: Callable[[Path], str | None] | None = None,
) -> None:
    """Raise (after unlinking `tmp`) unless the staged download looks complete.

    Call this immediately before the `tmp.replace(dest)` that publishes the file. Three
    independent checks, cheapest first:

    * `declared_length` — bytes received must equal `Content-Length` when the server sent
      one. This is the direct truncation test.
    * `min_bytes` — an absolute floor, for callers that already had one. Kept because it is
      free, but note it is **not** a truncation detector at any value: a 55 % truncation of
      the 1.31 GB `ESP_068483_2280` JP2 is still 719 MB and clears any floor below the
      149 MB smallest real file.
    * `validate` — a content check returning `None` when the file is acceptable, or a
      human-readable reason when it is not. This is the only check that survives a missing
      `Content-Length`, and the only one that catches a proxy padding to the declared size.

    `tmp` is removed on any failure, so a re-run starts clean rather than resuming onto a
    poisoned staging file.
    """
    tmp = Path(tmp)

    def _fail(reason: str) -> None:
        size = tmp.stat().st_size if tmp.exists() else -1
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"refusing to cache {url}: {reason} (staged file was {size} bytes; it has been "
            "removed, so re-running will retry cleanly)."
        )

    if not tmp.exists():
        raise RuntimeError(f"refusing to cache {url}: staged file {tmp} does not exist.")

    got = tmp.stat().st_size
    if declared_length is not None and got != declared_length:
        _fail(
            f"TRUNCATED download -- server declared Content-Length {declared_length:,} "
            f"but {got:,} bytes arrived ({declared_length - got:,} missing)"
        )
    if min_bytes is not None and got < min_bytes:
        _fail(f"only {got:,} bytes (< {min_bytes:,} floor); treating as malformed")
    if validate is not None:
        reason = validate(tmp)
        if reason is not None:
            _fail(reason)
