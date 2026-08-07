"""R66 — a truncated download must never reach a permanent cache.

`HTTPResponse.read(amt)` returns b"" on a premature EOF instead of raising, so every
stream-then-rename download path in this repo could publish a short file. For JP2 that is
especially bad: GDAL does not complain about a truncated JPEG2000 codestream, it silently
zero-fills the missing region, and Stage 2 reads that zero-fill as "no HiRISE coverage".

These tests are hermetic — a localhost HTTP server, synthetic payloads, tmp_path only.
See DECISIONS 2026-08-06t.
"""
from __future__ import annotations

import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from src import net
from src.hirise_imagery import ensure_jp2_local, inspect_jp2_integrity

# --------------------------------------------------------------------------------------
# A minimal JPEG2000 file builder. Real PDS JP2s use Lbox == 0 ("extends to EOF") for the
# jp2c box, which is precisely why a box walk alone cannot detect truncation — so the
# fixture must reproduce that, or the test would pass for the wrong reason.
# --------------------------------------------------------------------------------------
_SIG = bytes.fromhex("0000000C6A5020200D0A870A")


def _make_jp2(payload_len: int = 200, *, lbox_zero: bool = True) -> bytes:
    siz = b"\xff\x51" + (41).to_bytes(2, "big") + b"\x00" * 39
    # One tile-part: SOT(12) + payload + EOC(2). Psot spans SOT..end of tile-part data.
    psot = 12 + payload_len
    sot = (b"\xff\x90" + (10).to_bytes(2, "big") + (0).to_bytes(2, "big")
           + psot.to_bytes(4, "big") + b"\x00" + b"\x01")
    codestream = b"\xff\x4f" + siz + sot + b"\x5a" * payload_len + b"\xff\xd9"
    lbox = (0 if lbox_zero else 8 + len(codestream)).to_bytes(4, "big")
    return _SIG + lbox + b"jp2c" + codestream


def test_walker_accepts_a_complete_jp2(tmp_path):
    p = tmp_path / "ok.JP2"
    p.write_bytes(_make_jp2())
    out = inspect_jp2_integrity(p)
    assert out["status"] == "complete", out
    assert out["n_tile_parts"] == 1


def test_walker_accepts_an_explicit_lbox_too(tmp_path):
    p = tmp_path / "ok2.JP2"
    p.write_bytes(_make_jp2(lbox_zero=False))
    assert inspect_jp2_integrity(p)["status"] == "complete"


@pytest.mark.parametrize("drop", [1, 2, 3, 50, 199, 250])
def test_walker_detects_truncation_at_every_depth(tmp_path, drop):
    """Including the Lbox == 0 case, where the box walk alone can tell you nothing."""
    full = _make_jp2()
    p = tmp_path / "cut.JP2"
    p.write_bytes(full[:-drop])
    out = inspect_jp2_integrity(p)
    assert out["status"] == "truncated", out


def test_walker_reports_a_non_jp2_rather_than_raising(tmp_path):
    p = tmp_path / "notjp2.JP2"
    p.write_bytes(b"II*\x00" + b"\x00" * 5000)     # a TIFF, which the isolation suite stages
    assert inspect_jp2_integrity(p)["status"] == "not_jp2"


def test_walker_never_raises_on_junk(tmp_path):
    assert inspect_jp2_integrity(tmp_path / "missing.JP2")["status"] == "unreadable"
    empty = tmp_path / "empty.JP2"
    empty.write_bytes(b"")
    assert inspect_jp2_integrity(empty)["status"] == "not_jp2"


# --------------------------------------------------------------------------------------
# verify_download
# --------------------------------------------------------------------------------------

def test_verify_download_rejects_and_removes_a_short_file(tmp_path):
    tmp = tmp_path / "x.partial"
    tmp.write_bytes(b"0" * 100)
    with pytest.raises(RuntimeError, match="TRUNCATED"):
        net.verify_download(tmp, url="http://x/y", declared_length=500)
    assert not tmp.exists(), "the poisoned staging file must be removed"


def test_verify_download_passes_an_exact_length(tmp_path):
    tmp = tmp_path / "x.partial"
    tmp.write_bytes(b"0" * 500)
    net.verify_download(tmp, url="http://x/y", declared_length=500)
    assert tmp.exists()


def test_verify_download_survives_a_missing_content_length(tmp_path):
    tmp = tmp_path / "x.partial"
    tmp.write_bytes(b"0" * 500)
    net.verify_download(tmp, url="http://x/y", declared_length=None)


def test_verify_download_runs_the_content_validator(tmp_path):
    tmp = tmp_path / "x.partial"
    tmp.write_bytes(b"0" * 500)
    with pytest.raises(RuntimeError, match="bad content"):
        net.verify_download(tmp, url="http://x/y", validate=lambda p: "bad content")
    assert not tmp.exists()


# --------------------------------------------------------------------------------------
# End to end: a server that lies about Content-Length, i.e. a dropped connection.
# --------------------------------------------------------------------------------------

class _ShortHandler(BaseHTTPRequestHandler):
    payload = b""
    send_bytes = 0

    def do_GET(self):                                    # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload[: self.send_bytes])

    def log_message(self, *a):                           # keep pytest output clean
        pass


@pytest.fixture
def short_server():
    srv = HTTPServer(("127.0.0.1", 0), _ShortHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def test_a_dropped_jp2_download_is_not_committed(tmp_path, short_server):
    """The R66 headline: before the fix this published a short JP2 into the cache."""
    full = _make_jp2(payload_len=4_000_000)              # > the 1 MB floor when truncated
    _ShortHandler.payload = full
    _ShortHandler.send_bytes = 2_000_000                 # connection dies mid-transfer
    url = f"http://127.0.0.1:{short_server.server_port}/x.JP2"

    with pytest.raises(RuntimeError, match="TRUNCATED"):
        ensure_jp2_local("OBS_X", url, tmp_path)

    cached = tmp_path / "hirise_jp2" / "OBS_X_RED.JP2"
    assert not cached.exists(), "a short JP2 reached the permanent cache"
    assert not list((tmp_path / "hirise_jp2").glob("*.partial")), "staging file left behind"


def test_a_complete_jp2_download_is_committed(tmp_path, short_server):
    full = _make_jp2(payload_len=4_000_000)
    _ShortHandler.payload = full
    _ShortHandler.send_bytes = len(full)
    url = f"http://127.0.0.1:{short_server.server_port}/x.JP2"

    out = ensure_jp2_local("OBS_Y", url, tmp_path)
    assert out.exists() and out.read_bytes() == full
    assert inspect_jp2_integrity(out)["status"] == "complete"


def test_an_already_cached_truncated_jp2_raises_instead_of_being_reused(tmp_path):
    """The reuse gate was a 1 MB size floor, so a truncated cached file was preferred
    over /vsicurl/ forever. A 55 % truncation of the real 1.31 GB image is 719 MB and
    clears any floor below the 149 MB smallest genuine file."""
    cached = tmp_path / "hirise_jp2" / "OBS_Z_RED.JP2"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(_make_jp2(payload_len=4_000_000)[:-500])
    assert cached.stat().st_size > 1_000_000             # clears the old floor
    with pytest.raises(RuntimeError, match="TRUNCATED"):
        ensure_jp2_local("OBS_Z", "http://unused/", tmp_path)


def test_a_cached_non_jp2_is_still_accepted(tmp_path):
    """Lenient at reuse: tests/test_artifact_isolation.py deliberately stages a GeoTIFF
    under a `.JP2` name, and an unusual-but-legitimate JP2 must not be rejected by a
    walker that merely failed to parse it."""
    cached = tmp_path / "hirise_jp2" / "OBS_W_RED.JP2"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"II*\x00" + b"\x00" * 2_000_000)
    assert ensure_jp2_local("OBS_W", "http://unused/", tmp_path) == cached


# --------------------------------------------------------------------------------------
# The sibling paths that shared the identical hole.
# --------------------------------------------------------------------------------------

def test_ctx_tile_download_rejects_a_short_zip(tmp_path, short_server):
    from src.ctx_retrieve import _download_to

    buf = tmp_path / "src.zip"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.tif", b"\x00" * 60_000_000)
    _ShortHandler.payload = buf.read_bytes()
    _ShortHandler.send_bytes = len(_ShortHandler.payload) - 1000   # loses the central dir
    url = f"http://127.0.0.1:{short_server.server_port}/t.zip"

    dest = tmp_path / "out.zip"
    with pytest.raises(RuntimeError, match="TRUNCATED"):
        _download_to(url, dest)
    assert not dest.exists()


def test_validation_raster_download_rejects_a_short_file(tmp_path, short_server):
    from src.validation_retrieve import _download_raster

    _ShortHandler.payload = b"\x00" * 8_878_189
    _ShortHandler.send_bytes = 4_883_003          # the measured pre-fix commit
    url = f"http://127.0.0.1:{short_server.server_port}/r.tif"

    dest = tmp_path / "r.tif"
    with pytest.raises(RuntimeError, match="TRUNCATED"):
        _download_raster(url, dest)
    assert not dest.exists()


def test_every_live_cached_jp2_is_intact():
    """Regression guard over the real cache. Deliberately NOT marked slow: measured
    38.8 ms cold for all 46 files in one root, because it reads only marker headers."""
    repo = Path(__file__).resolve().parents[1]
    seen = 0
    for root in ("cache", "cache_v2"):
        d = repo / root / "hirise_jp2"
        if not d.exists():
            continue
        for p in sorted(d.glob("*.JP2")):
            seen += 1
            status = inspect_jp2_integrity(p)["status"]
            assert status != "truncated", f"{p} is TRUNCATED"
    if seen == 0:
        pytest.skip("no cached JP2s on this machine")
