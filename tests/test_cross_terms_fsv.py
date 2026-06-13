"""FSV for cross-terms materialisation (GitHub issue #20).

Source of truth: the persisted `cross_terms` table in a real sqlite-vec DB.
Happy path runs on the richest real project; edge cases use REAL synthetic
sqlite-vec databases with known instrument counts / missing windows.
"""
from __future__ import annotations

import json
import sqlite3
from math import comb
from pathlib import Path

import numpy as np
import pytest

from clipcannon.pipeline.cross_terms import (
    CrossTermError,
    materialise_cross_terms,
    rank_windows_by_coactivation,
)

PROJECTS = Path.home() / ".clipcannon" / "projects"
DIM = 6
WIN = 5000
ALL_INSTRUMENTS = ["vec_frames", "vec_semantic", "vec_emotion", "vec_speakers"]
_DDL = {
    "vec_frames": ("frame_id INTEGER PRIMARY KEY, project_id TEXT, timestamp_ms INTEGER, frame_path TEXT, visual_embedding", "timestamp_ms", "visual_embedding", "frame_id"),
    "vec_semantic": ("segment_id INTEGER PRIMARY KEY, project_id TEXT, timestamp_ms INTEGER, transcript_text TEXT, semantic_embedding", "timestamp_ms", "semantic_embedding", "segment_id"),
    "vec_emotion": ("id INTEGER PRIMARY KEY, project_id TEXT, start_ms INTEGER, end_ms INTEGER, emotion_embedding", "start_ms", "emotion_embedding", "id"),
    "vec_speakers": ("id INTEGER PRIMARY KEY, project_id TEXT, segment_text TEXT, timestamp_ms INTEGER, speaker_id INTEGER, speaker_embedding", "timestamp_ms", "speaker_embedding", "id"),
}


def _connect(path: Path) -> sqlite3.Connection:
    import sqlite_vec

    con = sqlite3.connect(str(path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def build_synth(
    path: Path,
    *,
    instruments: list[str],
    n_windows: int = 20,
    skip_window_for: tuple[str, int] | None = None,  # (instrument, window) to omit
    nan_for: tuple[str, int] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(path)
    con.execute("CREATE TABLE project (project_id TEXT)")
    con.execute("INSERT INTO project VALUES ('synth')")
    for tbl in instruments:
        cols, _, _, _ = _DDL[tbl]
        con.execute(f"CREATE VIRTUAL TABLE {tbl} USING vec0({cols} float[{DIM}])")
    rng = np.random.default_rng(3)
    for w in range(n_windows):
        ts = w * WIN + 1000
        for tbl in instruments:
            if skip_window_for and skip_window_for == (tbl, w):
                continue
            cols, tscol, vcol, pk = _DDL[tbl]
            vec = rng.normal(0, 1, DIM).tolist()
            if nan_for and nan_for == (tbl, w):
                vec[1] = float("nan")
            vj = json.dumps(vec)
            if tbl == "vec_frames":
                con.execute("INSERT INTO vec_frames(frame_id,project_id,timestamp_ms,frame_path,visual_embedding) VALUES (?,?,?,?,?)", (w + 1, "synth", ts, "", vj))
            elif tbl == "vec_semantic":
                con.execute("INSERT INTO vec_semantic(segment_id,project_id,timestamp_ms,transcript_text,semantic_embedding) VALUES (?,?,?,?,?)", (w + 1, "synth", ts, "", vj))
            elif tbl == "vec_emotion":
                con.execute("INSERT INTO vec_emotion(id,project_id,start_ms,end_ms,emotion_embedding) VALUES (?,?,?,?,?)", (w + 1, "synth", ts, ts + WIN, vj))
            elif tbl == "vec_speakers":
                con.execute("INSERT INTO vec_speakers(id,project_id,segment_text,timestamp_ms,speaker_id,speaker_embedding) VALUES (?,?,?,?,?,?)", (w + 1, "synth", "", ts, 1, vj))
    con.commit()
    con.close()
    return path


def _count(db: Path) -> int:
    con = sqlite3.connect(str(db))
    n = con.execute("SELECT COUNT(*) FROM cross_terms").fetchone()[0]
    con.close()
    return n


# --------------------------------------------------------------------------- #
# Happy path on real data
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_real_project_count_equals_windows_times_pairs():
    if not PROJECTS.is_dir():
        pytest.skip("no real projects (clean CI)")
    cand = sorted(PROJECTS.glob("proj_*"),
                  key=lambda p: -(p / "analysis.db").stat().st_size if (p / "analysis.db").exists() else 0)
    proj = next((p for p in cand if (p / "analysis.db").exists()), None)
    if proj is None:
        pytest.skip("no ingested project")
    stats = materialise_cross_terms(proj)
    total = _count(proj / "analysis.db")
    print(f"[FSV] {stats.project_id}: N={stats.n_instruments} C(N,2)={stats.n_pairs} "
          f"windows={stats.n_windows} total_rows={total} expected={stats.expected_rows} "
          f"(ok={stats.rows_written} missing={stats.rows_missing})")
    assert stats.n_pairs == comb(stats.n_instruments, 2)
    assert total == stats.expected_rows == stats.n_windows * stats.n_pairs
    assert stats.rows_written + stats.rows_missing == total


@pytest.mark.integration
def test_cross_term_lift_over_single_instrument():
    """#20: co-activation ranking must lift highlight discovery over a single
    instrument (measurable, on real data)."""
    if not PROJECTS.is_dir():
        pytest.skip("no real projects")
    proj = PROJECTS / "proj_2ea7221d"
    if not (proj / "analysis.db").exists():
        pytest.skip("reference project absent")
    materialise_cross_terms(proj)
    con = _connect(proj / "analysis.db")
    hl = set()
    for s, e in con.execute("SELECT start_ms, end_ms FROM highlights"):
        for w in range(s // WIN, e // WIN + 1):
            hl.add(w * WIN)
    k = 10
    co = [w for w, _ in rank_windows_by_coactivation(proj, k)]
    co_prec = sum(1 for w in co if w in hl) / k
    # visual-only baseline
    buck: dict[int, list] = {}
    for t, vj in con.execute("SELECT timestamp_ms, vec_to_json(visual_embedding) FROM vec_frames"):
        if t is None or vj is None:
            continue
        buck.setdefault((t // WIN) * WIN, []).append(np.array(json.loads(vj)))
    con.close()
    means = {w: np.mean(np.stack(v), 0) for w, v in buck.items()}
    keys = sorted(means)
    mat = np.stack([means[w] for w in keys])
    dev = np.linalg.norm(mat - mat.mean(0), axis=1)
    vis = [keys[i] for i in np.argsort(-dev)[:k]]
    vis_prec = sum(1 for w in vis if w in hl) / k
    print(f"[FSV] coactivation P@{k}={co_prec:.2f} vs visual-only P@{k}={vis_prec:.2f} "
          f"lift={co_prec - vis_prec:+.2f}")
    assert co_prec >= vis_prec, "cross-term co-activation must not underperform a single instrument"
    assert co_prec > 0


# --------------------------------------------------------------------------- #
# Edge cases (synthetic real DBs)
# --------------------------------------------------------------------------- #
def test_edge_n_varies_pairs_track_comb(tmp_path):
    """#20 edge 2: C(N,2) must update with N automatically (2->1, 3->3, 4->6)."""
    for n in (2, 3, 4):
        insts = ALL_INSTRUMENTS[:n]
        db = build_synth(tmp_path / f"n{n}" / "analysis.db", instruments=insts, n_windows=12)
        stats = materialise_cross_terms(tmp_path / f"n{n}")
        print(f"[FSV] N={n} -> pairs={stats.n_pairs} rows={_count(db)}")
        assert stats.n_pairs == comb(n, 2)
        assert _count(db) == stats.n_windows * comb(n, 2)


def test_edge_missing_instrument_window_flagged_not_zero(tmp_path):
    """#20 edge 1: a window missing one instrument -> rows flagged 'missing',
    never silently written as 0."""
    db = build_synth(
        tmp_path / "miss" / "analysis.db",
        instruments=ALL_INSTRUMENTS,
        n_windows=15,
        skip_window_for=("vec_emotion", 7),
    )
    materialise_cross_terms(tmp_path / "miss")
    con = sqlite3.connect(str(db))
    # window 7 (ms=36000) pairs involving emotion must be 'missing' with NULL value
    miss = con.execute(
        "SELECT instrument_i, instrument_j, value, status FROM cross_terms "
        "WHERE window_ms=? AND (instrument_i LIKE '%emotion%' OR instrument_j LIKE '%emotion%')",
        (7 * WIN,),
    ).fetchall()
    con.close()
    print(f"[FSV] window-7 emotion pairs: {miss}")
    assert miss, "no cross-term rows for the skipped window"
    for _i, _j, value, status in miss:
        assert status == "missing" and value is None, "missing data was silently zero-filled"


def test_edge_empty_project_zero_cross_terms(tmp_path):
    """#20 edge 4: empty project -> 0 windows -> 0 cross-terms, no crash."""
    db = build_synth(tmp_path / "empty" / "analysis.db", instruments=ALL_INSTRUMENTS, n_windows=0)
    stats = materialise_cross_terms(tmp_path / "empty")
    assert stats.n_windows == 0
    assert _count(db) == 0


def test_edge_nan_embedding_raises(tmp_path):
    """NaN in a source vector must raise (no silent corrupt cross-term)."""
    build_synth(
        tmp_path / "nan" / "analysis.db",
        instruments=ALL_INSTRUMENTS,
        n_windows=10,
        nan_for=("vec_speakers", 4),
    )
    with pytest.raises(CrossTermError) as ei:
        materialise_cross_terms(tmp_path / "nan")
    assert "NaN" in str(ei.value) and "vec_speakers" in str(ei.value)


def test_idempotent_remateralise(tmp_path):
    """Re-running must not double-count (DELETE-then-insert)."""
    db = build_synth(tmp_path / "idem" / "analysis.db", instruments=ALL_INSTRUMENTS, n_windows=10)
    s1 = materialise_cross_terms(tmp_path / "idem")
    n1 = _count(db)
    materialise_cross_terms(tmp_path / "idem")
    n2 = _count(db)
    assert n1 == n2 == s1.n_windows * s1.n_pairs
