"""FSV for the instrument differentiation-contract validator (GitHub #19).

Source of truth:
  * happy path — the real ingested projects under ~/.clipcannon/projects
    (integration; skips cleanly in clean CI).
  * 5 edge cases — REAL sqlite-vec databases we build with known, controlled
    contents (synthetic inputs with known expected outputs, not mocks). Each
    proves the validator FAILS LOUD on a specific broken state.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from clipcannon.pipeline.differentiation import (
    DifferentiationError,
    validate_differentiation,
)

PROJECTS = Path.home() / ".clipcannon" / "projects"
DIM = 6  # small synthetic embedding dim (the validator is dim-agnostic)


def _connect(path: Path) -> sqlite3.Connection:
    import sqlite_vec

    con = sqlite3.connect(str(path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def build_db(
    path: Path,
    *,
    n_windows: int = 40,
    degenerate: str | None = None,   # instrument -> all-zero vectors
    duplicate: tuple[str, str] | None = None,  # (dst, src) identical vectors
    nan_in: str | None = None,       # instrument -> inject a NaN
    constant_outcome: bool = False,  # one scene, one sentiment, one speaker, no highlight
    n_speakers: int = 2,
) -> Path:
    """Create a real sqlite-vec analysis.db with controlled, known contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(path)
    con.executescript(
        f"""
        CREATE VIRTUAL TABLE vec_frames USING vec0(
            frame_id INTEGER PRIMARY KEY, project_id TEXT, timestamp_ms INTEGER,
            frame_path TEXT, visual_embedding float[{DIM}]);
        CREATE VIRTUAL TABLE vec_semantic USING vec0(
            segment_id INTEGER PRIMARY KEY, project_id TEXT, timestamp_ms INTEGER,
            transcript_text TEXT, semantic_embedding float[{DIM}]);
        CREATE VIRTUAL TABLE vec_emotion USING vec0(
            id INTEGER PRIMARY KEY, project_id TEXT, start_ms INTEGER, end_ms INTEGER,
            emotion_embedding float[{DIM}]);
        CREATE VIRTUAL TABLE vec_speakers USING vec0(
            id INTEGER PRIMARY KEY, project_id TEXT, segment_text TEXT,
            timestamp_ms INTEGER, speaker_id INTEGER, speaker_embedding float[{DIM}]);
        CREATE TABLE highlights (start_ms INTEGER, end_ms INTEGER, score REAL);
        CREATE TABLE scenes (scene_id INTEGER, start_ms INTEGER, end_ms INTEGER, shot_type TEXT);
        CREATE TABLE transcript_segments (start_ms INTEGER, end_ms INTEGER,
            sentiment TEXT, speaker_id INTEGER);
        """
    )
    rng = np.random.default_rng(7)
    win_ms = 5000

    cache: dict[tuple[str, int], list[float]] = {}

    def make_vec(inst: str, driver: float) -> list[float]:
        if degenerate == inst:
            return [0.0] * DIM
        # dim 0 encodes this instrument's OWN driver (its grounded outcome);
        # independent gaussian noise in the rest keeps cross-instrument corr low.
        base = rng.normal(0, 0.1, DIM)
        base[0] += float(driver)
        return base.tolist()

    for w in range(n_windows):
        ts = w * win_ms + 1000
        # Independent drivers so each instrument grounds a DIFFERENT outcome and
        # the instruments are mutually non-redundant.
        scene = 0 if constant_outcome else (w // 8)            # frames driver
        sent_bit = 0 if constant_outcome else (w // 3) % 2     # semantic driver
        hl_flag = 0 if constant_outcome else int(w % 6 == 0)   # emotion driver
        speaker = 1 if constant_outcome else (w % n_speakers) + 1  # speaker driver
        shot = "closeup" if constant_outcome else ("closeup" if (w // 8) % 2 else "wide")
        sentiment = "POSITIVE" if sent_bit else "NEGATIVE"

        drivers = {
            "vec_frames": scene,
            "vec_semantic": sent_bit,
            "vec_emotion": hl_flag,
            "vec_speakers": speaker,
        }
        # order matters: fill 'src' instruments before a duplicate copies them.
        for inst in ("vec_frames", "vec_semantic", "vec_emotion", "vec_speakers"):
            if duplicate and duplicate[0] == inst:
                cache[(inst, w)] = list(cache[(duplicate[1], w)])
            else:
                cache[(inst, w)] = make_vec(inst, drivers[inst])
        fv = json.dumps(cache[("vec_frames", w)])
        sv = json.dumps(cache[("vec_semantic", w)])
        ev_list = cache[("vec_emotion", w)]
        if nan_in == "vec_emotion" and w == 3:
            ev_list = list(ev_list)
            ev_list[2] = float("nan")
        ev = json.dumps(ev_list)
        kv = json.dumps(cache[("vec_speakers", w)])
        con.execute("INSERT INTO vec_frames(frame_id,project_id,timestamp_ms,frame_path,visual_embedding) VALUES (?,?,?,?,?)",
                    (w + 1, "synth", ts, "", fv))
        con.execute("INSERT INTO vec_semantic(segment_id,project_id,timestamp_ms,transcript_text,semantic_embedding) VALUES (?,?,?,?,?)",
                    (w + 1, "synth", ts, "", sv))
        con.execute("INSERT INTO vec_emotion(id,project_id,start_ms,end_ms,emotion_embedding) VALUES (?,?,?,?,?)",
                    (w + 1, "synth", ts, ts + win_ms, ev))
        con.execute("INSERT INTO vec_speakers(id,project_id,segment_text,timestamp_ms,speaker_id,speaker_embedding) VALUES (?,?,?,?,?,?)",
                    (w + 1, "synth", "", ts, speaker, kv))
        con.execute("INSERT INTO scenes(scene_id,start_ms,end_ms,shot_type) VALUES (?,?,?,?)",
                    (scene, ts, ts + win_ms, shot))
        con.execute("INSERT INTO transcript_segments(start_ms,end_ms,sentiment,speaker_id) VALUES (?,?,?,?)",
                    (ts, ts + win_ms, sentiment, speaker))
        if not constant_outcome and w % 6 == 0:
            con.execute("INSERT INTO highlights(start_ms,end_ms,score) VALUES (?,?,?)",
                        (ts, ts + win_ms, 1.0))
    con.commit()
    con.close()
    return path


# --------------------------------------------------------------------------- #
# Happy path — real corpus
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_real_corpus_passes_contract():
    if not PROJECTS.is_dir():
        pytest.skip("no real projects (clean CI)")
    dirs = [p for p in PROJECTS.glob("proj_*") if (p / "analysis.db").exists()]
    # need enough projects/windows; pick those that actually have embeddings
    dirs = sorted(dirs)
    if len(dirs) < 2:
        pytest.skip("need >=2 ingested projects")
    report = validate_differentiation(dirs)
    print(report.render())
    # Every pair must be below the redundancy ceiling on real data.
    assert all(c <= 0.6 for c in report.corr.values()), "real instruments are redundant!"
    # At least the visual/semantic/emotion instruments must be grounded.
    assert report.status.get("vec_frames") == "OK"
    assert report.bits["vec_frames"] >= 0.05


# --------------------------------------------------------------------------- #
# Edge cases (synthetic real DBs)
# --------------------------------------------------------------------------- #
def test_edge1_zero_vector_instrument_fails(tmp_path):
    """#19 edge 1: a zero-vector (degenerate) emotion instrument must FAIL."""
    build_db(tmp_path / "p1" / "analysis.db", degenerate="vec_emotion")
    rep = validate_differentiation([tmp_path / "p1"])
    print(rep.render())
    assert not rep.passed
    assert rep.status["vec_emotion"] == "DEGENERATE"
    assert any("vec_emotion" in f and "degenerate" in f.lower() for f in rep.failures)


def test_edge2_duplicate_instrument_fails_correlation(tmp_path):
    """#19 edge 2: duplicating an instrument must trip the >0.6 pair check."""
    # emotion becomes an exact copy of semantic -> their PC1 correlate at 1.0.
    build_db(tmp_path / "p2" / "analysis.db", duplicate=("vec_emotion", "vec_semantic"))
    rep = validate_differentiation([tmp_path / "p2"])
    print(rep.render())
    assert not rep.passed
    pair = rep.corr.get(("vec_emotion", "vec_semantic")) or rep.corr.get(("vec_semantic", "vec_emotion"))
    assert pair is not None and pair > 0.6, f"duplicate pair corr={pair}"


def test_edge3_single_tiny_project_refuses_to_certify(tmp_path):
    """#19 edge 3: too few aligned windows -> refuse (DifferentiationError)."""
    build_db(tmp_path / "p3" / "analysis.db", n_windows=3)
    with pytest.raises(DifferentiationError) as ei:
        validate_differentiation([tmp_path / "p3"])
    print(f"[FSV] {ei.value}")
    assert "windows" in str(ei.value).lower()


def test_edge4_constant_outcome_errors(tmp_path):
    """#19 edge 4: all outcomes identical -> MI undefined -> must error."""
    build_db(tmp_path / "p4" / "analysis.db", constant_outcome=True)
    with pytest.raises(DifferentiationError) as ei:
        validate_differentiation([tmp_path / "p4"])
    print(f"[FSV] {ei.value}")
    assert "outcome" in str(ei.value).lower() or "undefined" in str(ei.value).lower()


def test_edge5_nan_embedding_is_surfaced(tmp_path):
    """#19 edge 5: a NaN in an embedding must be surfaced with project/row, not crash."""
    build_db(tmp_path / "p5" / "analysis.db", nan_in="vec_emotion")
    with pytest.raises(DifferentiationError) as ei:
        validate_differentiation([tmp_path / "p5"])
    msg = str(ei.value)
    print(f"[FSV] {msg}")
    assert "NaN" in msg and "vec_emotion" in msg and "rowid" in msg


def test_healthy_synthetic_corpus_passes(tmp_path):
    """Control: a well-formed synthetic corpus PASSES (the test isn't vacuous)."""
    build_db(tmp_path / "ok" / "analysis.db", n_windows=48, n_speakers=2)
    rep = validate_differentiation([tmp_path / "ok"])
    print(rep.render())
    assert rep.passed, rep.failures
    assert all(c <= 0.6 for c in rep.corr.values())
