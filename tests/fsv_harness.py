"""Reusable Full State Verification (FSV) harness (GitHub issue #57).

FSV means: never trust a function's return value — go read the **source of
truth** (the database row, the vector blob, the file on disk) with an
INDEPENDENT reader and prove the data is actually there and correct.

Design rules:
- Reads use RAW sqlite3 / ffprobe, NOT the application's own DB helpers. If we
  verified the app with the app's code, a bug in that code could hide itself.
- Every assert prints the ACTUAL observed state (evidence), not just pass/fail.
- A helper that cannot find its artifact RAISES — it never returns a vacuous
  "pass". Absence of evidence is a failure, not a success.

These primitives are imported by the per-issue FSV tests.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - sqlite_vec is a hard dep for vec reads
    sqlite_vec = None


# --------------------------------------------------------------------------- #
# Evidence printing
# --------------------------------------------------------------------------- #
def evidence(label: str, value: Any) -> None:
    """Print actual observed state as evidence of success/failure."""
    print(f"[FSV] {label}: {value}")


# --------------------------------------------------------------------------- #
# Independent SQLite reader (raw — does NOT use clipcannon.db helpers)
# --------------------------------------------------------------------------- #
def _connect(db_path: str | Path, load_vec: bool = False) -> sqlite3.Connection:
    """Open a raw, independent read connection to the source of truth."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Source-of-truth DB does not exist: {p}")
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    if load_vec:
        if sqlite_vec is None:
            raise RuntimeError(
                "sqlite_vec is not installed but a vector table read was "
                "requested — cannot verify the vector source of truth."
            )
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    return conn


def raw_query(
    db_path: str | Path,
    sql: str,
    params: tuple = (),
    *,
    load_vec: bool = False,
) -> list[sqlite3.Row]:
    """Run a read query against the source of truth and return the rows."""
    conn = _connect(db_path, load_vec=load_vec)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def table_rowcount(
    db_path: str | Path,
    table: str,
    where: str = "",
    params: tuple = (),
    *,
    load_vec: bool = False,
) -> int:
    """Count rows in a table (optionally filtered) — the actual stored count."""
    clause = f" WHERE {where}" if where else ""
    rows = raw_query(db_path, f"SELECT COUNT(*) AS n FROM {table}{clause}", params, load_vec=load_vec)
    return int(rows[0]["n"])


def assert_rowcount(
    db_path: str | Path,
    table: str,
    expected: int,
    where: str = "",
    params: tuple = (),
    *,
    load_vec: bool = False,
) -> int:
    """Assert a table holds exactly `expected` rows; print the observed count."""
    actual = table_rowcount(db_path, table, where, params, load_vec=load_vec)
    evidence(f"{table} rowcount" + (f" WHERE {where}" if where else ""), actual)
    if actual != expected:
        raise AssertionError(
            f"{table}: expected {expected} rows, source of truth holds {actual}"
            + (f" (WHERE {where})" if where else "")
        )
    return actual


# --------------------------------------------------------------------------- #
# Vector-store verification (sqlite-vec vec0 tables)
# --------------------------------------------------------------------------- #
def read_vectors(
    db_path: str | Path,
    table: str,
    vec_col: str,
    dim: int,
    where: str = "",
    params: tuple = (),
) -> np.ndarray:
    """Read a vec0 vector column back as an (N, dim) float32 array.

    Unpacks the stored little-endian float32 blobs directly from the source of
    truth. Raises if any blob's length does not match `dim`.
    """
    clause = f" WHERE {where}" if where else ""
    rows = raw_query(
        db_path,
        f"SELECT {vec_col} AS v FROM {table}{clause}",
        params,
        load_vec=True,
    )
    vecs = []
    for i, r in enumerate(rows):
        blob = r["v"]
        vec = np.frombuffer(blob, dtype="<f4")
        if vec.shape[0] != dim:
            raise AssertionError(
                f"{table}.{vec_col} row {i}: stored dim {vec.shape[0]} != expected {dim}"
            )
        vecs.append(vec)
    if not vecs:
        return np.empty((0, dim), dtype=np.float32)
    return np.stack(vecs)


def assert_vector_store(
    db_path: str | Path,
    table: str,
    vec_col: str,
    dim: int,
    *,
    min_count: int = 1,
    require_nonzero: bool = True,
    require_variance: bool = True,
    where: str = "",
    params: tuple = (),
) -> np.ndarray:
    """Verify a vector store holds non-degenerate vectors of the right shape.

    Catches the dead-instrument signatures: missing rows, wrong dim, all-zero
    vectors, or zero variance across rows (a constant embedding).
    """
    arr = read_vectors(db_path, table, vec_col, dim, where, params)
    evidence(f"{table}.{vec_col} shape", arr.shape)
    if arr.shape[0] < min_count:
        raise AssertionError(
            f"{table}.{vec_col}: only {arr.shape[0]} vectors, expected >= {min_count}"
        )
    assert_no_nan(arr, f"{table}.{vec_col}")
    if require_nonzero and not np.any(arr):
        raise AssertionError(
            f"{table}.{vec_col}: ALL vectors are zero — degenerate / dead instrument"
        )
    if require_variance and arr.shape[0] >= 2:
        var = float(np.var(arr))
        evidence(f"{table}.{vec_col} variance", round(var, 8))
        if var == 0.0:
            raise AssertionError(
                f"{table}.{vec_col}: zero variance across {arr.shape[0]} vectors "
                "(constant embedding — instrument carries 0 bits)"
            )
    return arr


def assert_no_nan(arr: np.ndarray, label: str) -> None:
    """Assert an array contains no NaN/Inf (silent corruption)."""
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.count_nonzero(~np.isfinite(arr)))
        raise AssertionError(f"{label}: contains {n_nan} non-finite (NaN/Inf) values")


# --------------------------------------------------------------------------- #
# Media verification (ffprobe — independent of any app render code)
# --------------------------------------------------------------------------- #
def ffprobe_media(path: str | Path) -> dict:
    """Probe a media file with ffprobe and return format+stream metadata."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Media source of truth does not exist: {p}")
    if p.stat().st_size == 0:
        raise AssertionError(f"Media file is empty (0 bytes): {p}")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_format", "-show_streams",
            "-of", "json", str(p),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise AssertionError(f"ffprobe failed on {p}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def assert_media(
    path: str | Path,
    *,
    width: int | None = None,
    height: int | None = None,
    codec: str | None = None,
    min_duration_s: float | None = None,
    sample_rate: int | None = None,
    has_audio: bool | None = None,
) -> dict:
    """Assert a media file's actual properties match expectations (via ffprobe)."""
    meta = ffprobe_media(path)
    streams = meta.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(meta.get("format", {}).get("duration", 0.0))
    evidence(
        Path(path).name,
        {
            "video": (video.get("width"), video.get("height"), video.get("codec_name")) if video else None,
            "audio": (audio.get("sample_rate"), audio.get("channels")) if audio else None,
            "duration_s": round(dur, 3),
        },
    )
    if width is not None and (not video or int(video["width"]) != width):
        raise AssertionError(f"{path}: width {video and video.get('width')} != {width}")
    if height is not None and (not video or int(video["height"]) != height):
        raise AssertionError(f"{path}: height {video and video.get('height')} != {height}")
    if codec is not None and (not video or video.get("codec_name") != codec):
        raise AssertionError(f"{path}: codec {video and video.get('codec_name')} != {codec}")
    if min_duration_s is not None and dur < min_duration_s:
        raise AssertionError(f"{path}: duration {dur:.3f}s < {min_duration_s}s")
    if sample_rate is not None and (not audio or int(audio["sample_rate"]) != sample_rate):
        raise AssertionError(f"{path}: sample_rate {audio and audio.get('sample_rate')} != {sample_rate}")
    if has_audio is True and audio is None:
        raise AssertionError(f"{path}: expected an audio stream, found none")
    if has_audio is False and audio is not None:
        raise AssertionError(f"{path}: expected NO audio stream, found one")
    return meta


# --------------------------------------------------------------------------- #
# Before/after state snapshots (prove a trigger changed the source of truth)
# --------------------------------------------------------------------------- #
def snapshot(db_path: str | Path, tables: list[str], *, load_vec: bool = False) -> dict[str, int]:
    """Snapshot rowcounts for a set of tables (the state before/after a trigger)."""
    snap = {}
    for t in tables:
        try:
            snap[t] = table_rowcount(db_path, t, load_vec=load_vec)
        except sqlite3.OperationalError as exc:
            snap[t] = -1  # table missing — recorded explicitly, not hidden
            evidence(f"snapshot {t}", f"MISSING ({exc})")
    return snap


def diff_snapshot(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Return the per-table rowcount delta and print it as evidence."""
    delta = {t: after.get(t, 0) - before.get(t, 0) for t in set(before) | set(after)}
    evidence("state delta (after - before)", delta)
    return delta
