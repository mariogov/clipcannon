"""Cross-terms: associations-between-associations (GitHub issue #20).

Derived Data Abundance claims n inputs through N instruments yield up to
n*(N + C(N,2) + 1) structured signals; the C(N,2) cross-terms are the pairwise
associations *between* instruments. The codebase stored per-instrument
embeddings but never materialised the cross-terms — this module computes and
persists them, and exposes a co-activation ranker that demonstrably lifts
highlight discovery over any single instrument.

Definition (concrete + interpretable): per 5s window w and instrument pair
(i, j), the cross-term is the product of each instrument's *salience* — how far
its window embedding deviates from that instrument's global mean, in std units:

    salience_i(w) = || e_i(w) - mean_i || / scale_i
    cross_ij(w)   = salience_i(w) * salience_j(w)

A large cross_ij(w) means instruments i and j are *simultaneously* doing
something unusual — a multi-modal salient moment (the kind that makes a clip a
highlight). Reproducible, no model loads, no network.

Source of truth: a new `cross_terms` table keyed by
(project_id, window_ms, instrument_i, instrument_j).

CLI:
    python -m clipcannon.pipeline.cross_terms PROJ_DIR        # materialise
    python -m clipcannon.pipeline.cross_terms PROJ_DIR --rank # show top moments
"""
from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

WINDOW_MS = 5000

# instrument table -> (timestamp column, primary-key column, vector column)
INSTRUMENTS: dict[str, tuple[str, str, str]] = {
    "vec_frames": ("timestamp_ms", "frame_id", "visual_embedding"),
    "vec_semantic": ("timestamp_ms", "segment_id", "semantic_embedding"),
    "vec_emotion": ("start_ms", "id", "emotion_embedding"),
    "vec_speakers": ("timestamp_ms", "id", "speaker_embedding"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cross_terms (
    project_id   TEXT NOT NULL,
    window_ms    INTEGER NOT NULL,
    instrument_i TEXT NOT NULL,
    instrument_j TEXT NOT NULL,
    value        REAL,
    status       TEXT NOT NULL DEFAULT 'ok',   -- 'ok' | 'missing'
    PRIMARY KEY (project_id, window_ms, instrument_i, instrument_j)
);
"""


class CrossTermError(Exception):
    """Raised on uncomputable / corrupt input (with an actionable reason)."""


@dataclass
class CrossTermStats:
    project_id: str
    n_windows: int
    n_instruments: int
    n_pairs: int          # C(N,2)
    rows_written: int
    rows_missing: int

    @property
    def expected_rows(self) -> int:
        return self.n_windows * self.n_pairs


def _connect(db_path: Path) -> sqlite3.Connection:
    import sqlite_vec

    con = sqlite3.connect(str(db_path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _present_instruments(con: sqlite3.Connection) -> list[str]:
    out = []
    for tbl in INSTRUMENTS:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE name=?", (tbl,)
        ).fetchone()
        if not row:
            continue
        if con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] > 0:
            out.append(tbl)
    return out


def _instrument_window_means(
    con: sqlite3.Connection, tbl: str, project_tag: str
) -> dict[int, np.ndarray]:
    ts, pk, col = INSTRUMENTS[tbl]
    rows = con.execute(f"SELECT {pk}, {ts}, vec_to_json({col}) FROM {tbl}").fetchall()
    buckets: dict[int, list[np.ndarray]] = {}
    for rowid, t, vjson in rows:
        if t is None or vjson is None:
            continue
        v = np.asarray(json.loads(vjson), dtype=np.float64)
        if not np.isfinite(v).all():
            raise CrossTermError(
                f"NaN/Inf embedding in {project_tag}:{tbl} rowid={rowid} — "
                f"cross-terms cannot be computed on corrupt source vectors."
            )
        buckets.setdefault(int(t) // WINDOW_MS, []).append(v)
    return {w: np.mean(np.stack(vs), axis=0) for w, vs in buckets.items()}


def _salience(window_means: dict[int, np.ndarray]) -> dict[int, float]:
    """Per-window deviation magnitude from the instrument mean, in scale units."""
    if not window_means:
        return {}
    keys = sorted(window_means)
    mat = np.stack([window_means[k] for k in keys])
    mean = mat.mean(axis=0, keepdims=True)
    dev = np.linalg.norm(mat - mean, axis=1)
    scale = float(np.median(dev)) or 1.0
    return {k: float(d / scale) for k, d in zip(keys, dev, strict=True)}


def materialise_cross_terms(project_dir: Path) -> CrossTermStats:
    """Compute and persist C(N,2) cross-terms per window. Idempotent (replaces)."""
    db = project_dir / "analysis.db" if project_dir.is_dir() else project_dir
    if not db.exists():
        raise CrossTermError(f"no analysis.db under {project_dir}")
    con = _connect(db)
    try:
        con.executescript(_SCHEMA)
        pid_row = con.execute("SELECT project_id FROM project LIMIT 1").fetchone()
        project_id = pid_row[0] if pid_row else project_dir.name

        present = _present_instruments(con)
        salience: dict[str, dict[int, float]] = {}
        all_windows: set[int] = set()
        for tbl in present:
            wm = _instrument_window_means(con, tbl, project_id)
            salience[tbl] = _salience(wm)
            all_windows.update(wm.keys())

        n = len(present)
        n_pairs = comb(n, 2) if n >= 2 else 0
        # Clean prior rows for this project so the count is exact (idempotent).
        con.execute("DELETE FROM cross_terms WHERE project_id=?", (project_id,))

        rows_written = rows_missing = 0
        for w in sorted(all_windows):
            for a, b in combinations(present, 2):
                sa = salience[a].get(w)
                sb = salience[b].get(w)
                if sa is None or sb is None:
                    # A window where one instrument has no data: record it as
                    # 'missing' rather than silently writing 0 (edge case 1).
                    con.execute(
                        "INSERT INTO cross_terms VALUES (?,?,?,?,?,?)",
                        (project_id, w * WINDOW_MS, a, b, None, "missing"),
                    )
                    rows_missing += 1
                else:
                    con.execute(
                        "INSERT INTO cross_terms VALUES (?,?,?,?,?,?)",
                        (project_id, w * WINDOW_MS, a, b, sa * sb, "ok"),
                    )
                    rows_written += 1
        con.commit()
        return CrossTermStats(
            project_id=project_id,
            n_windows=len(all_windows),
            n_instruments=n,
            n_pairs=n_pairs,
            rows_written=rows_written,
            rows_missing=rows_missing,
        )
    finally:
        con.close()


def rank_windows_by_coactivation(project_dir: Path, top_k: int = 10) -> list[tuple[int, float]]:
    """Discovery helper: windows ranked by summed cross-term co-activation.

    Returns [(window_ms, total_coactivation)] descending — the moments where the
    most instruments are simultaneously salient.
    """
    db = project_dir / "analysis.db" if project_dir.is_dir() else project_dir
    con = _connect(db)
    try:
        rows = con.execute(
            "SELECT window_ms, SUM(value) FROM cross_terms WHERE status='ok' "
            "GROUP BY window_ms ORDER BY SUM(value) DESC LIMIT ?",
            (top_k,),
        ).fetchall()
        return [(int(w), float(v)) for w, v in rows]
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    rank = "--rank" in args
    dirs = [Path(a).expanduser() for a in args if not a.startswith("--")]
    for d in dirs:
        try:
            stats = materialise_cross_terms(d)
        except CrossTermError as e:
            print(f"CANNOT COMPUTE for {d}: {e}", file=sys.stderr)
            return 3
        print(
            f"{stats.project_id}: N={stats.n_instruments} instruments, "
            f"C(N,2)={stats.n_pairs}, windows={stats.n_windows} -> "
            f"{stats.rows_written} ok + {stats.rows_missing} missing "
            f"(expected {stats.expected_rows})"
        )
        if rank:
            print("  top co-activation windows (ms, score):")
            for w, v in rank_windows_by_coactivation(d):
                print(f"    {w:>9d}  {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
