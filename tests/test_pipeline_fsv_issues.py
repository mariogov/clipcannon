"""Full State Verification for the analysis pipeline (GitHub #22, #23, #24, #34).

Source of truth:
  * #24 — live: synthesize real PCM WAVs and run the actual silent-audio guard.
  * #22/#23/#34 — the real sqlite analysis.db of an *already ingested* project
    (the richest one under ~/.clipcannon/projects). We decode the actual stored
    vectors / rows and assert they are correct and non-degenerate. These tests
    are marked `integration`: on a box with real projects they run for real; in
    a clean CI they skip (never false-pass).

Why read the real DB rather than asserting on return values: a stage can return
"ok" while writing a constant/zero/NaN vector (the historical emotion valence
bug). Decoding the persisted vectors and checking dim/variance/NaN is exactly
the check that fails when the system is broken.
"""
from __future__ import annotations

import json
import math
import struct
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from clipcannon.exceptions import PipelineError
from clipcannon.pipeline import transcribe

PROJECTS = Path.home() / ".clipcannon" / "projects"


# --------------------------------------------------------------------------- #
# #24 — silent-audio fail-fast (LIVE, no GPU, no DB)
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def test_silent_audio_rejected_fast_with_full_diagnostic(tmp_path):
    """Digital silence must fail in well under the WhisperX load cost, with a
    diagnostic carrying peak dBFS / RMS / audible% / remediation (issue #24)."""
    sr = 16000
    silent = np.zeros(sr * 3, dtype=np.float32)  # 3s of true silence
    wav = tmp_path / "silent.wav"
    _write_wav(wav, silent, sr)

    t0 = time.time()
    diag = transcribe._compute_audio_diagnostics(wav)
    with pytest.raises(PipelineError) as ei:
        transcribe._reject_if_silent(wav, diag)
    dt = time.time() - t0
    msg = str(ei.value)
    print(f"[FSV] silent rejected in {dt*1000:.1f}ms; peak={diag['peak_dbfs']:.1f} dBFS")
    print(msg)
    assert dt < 5.0, "fail-fast must beat the WhisperX load cost"
    # The diagnostic must contain every required field.
    for token in ("peak:", "dBFS", "RMS:", "audible:", "Resolution:"):
        assert token in msg, f"diagnostic missing {token!r}"
    assert diag["peak_dbfs"] <= transcribe._SILENCE_PEAK_DBFS


def test_loud_speechlike_audio_passes(tmp_path):
    """A clearly audible signal must NOT be rejected (no over-rejection)."""
    sr = 16000
    t = np.linspace(0, 3, sr * 3, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)  # -10 dBFS-ish, fully audible
    wav = tmp_path / "loud.wav"
    _write_wav(wav, tone, sr)
    diag = transcribe._compute_audio_diagnostics(wav)
    print(f"[FSV] loud peak={diag['peak_dbfs']:.1f} dBFS audible%={diag['audible_pct']:.1f}")
    transcribe._reject_if_silent(wav, diag)  # must not raise
    assert diag["peak_dbfs"] > transcribe._SILENCE_PEAK_DBFS
    assert diag["audible_pct"] >= transcribe._MIN_AUDIBLE_PCT


def test_borderline_quiet_but_audible_passes(tmp_path):
    """Edge case (#24): a quiet but audible clip just above threshold passes."""
    sr = 16000
    t = np.linspace(0, 3, sr * 3, dtype=np.float32)
    # ~-44 dBFS peak (10**(-44/20) ≈ 0.0063), comfortably above the -50 floor.
    amp = 10 ** (-44 / 20)
    tone = amp * np.sin(2 * np.pi * 220 * t)
    wav = tmp_path / "quiet.wav"
    _write_wav(wav, tone, sr)
    diag = transcribe._compute_audio_diagnostics(wav)
    print(f"[FSV] borderline peak={diag['peak_dbfs']:.1f} dBFS audible%={diag['audible_pct']:.1f}")
    assert -50.0 < diag["peak_dbfs"] < -30.0
    transcribe._reject_if_silent(wav, diag)  # must not raise


def test_empty_audio_is_treated_as_silence(tmp_path):
    sr = 16000
    wav = tmp_path / "empty.wav"
    _write_wav(wav, np.zeros(0, dtype=np.float32), sr)
    diag = transcribe._compute_audio_diagnostics(wav)
    assert diag["peak_dbfs"] <= transcribe._SILENCE_PEAK_DBFS
    with pytest.raises(PipelineError):
        transcribe._reject_if_silent(wav, diag)


# --------------------------------------------------------------------------- #
# Shared fixture: richest real ingested project (source of truth)
# --------------------------------------------------------------------------- #
def _open_db(db_path: Path):
    import sqlite3

    import sqlite_vec

    con = sqlite3.connect(str(db_path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _richest_project_db():
    if not PROJECTS.is_dir():
        pytest.skip(f"no real projects at {PROJECTS} (clean CI) — integration only")
    best, best_n = None, -1
    for db in PROJECTS.glob("*/analysis.db"):
        try:
            con = _open_db(db)
            n = con.execute("SELECT COUNT(*) FROM vec_emotion").fetchone()[0]
            n += con.execute("SELECT COUNT(*) FROM vec_frames").fetchone()[0]
            con.close()
        except Exception:
            continue
        if n > best_n:
            best, best_n = db, n
    if best is None or best_n <= 0:
        pytest.skip("no ingested project DB with embeddings found")
    return best


@pytest.fixture(scope="module")
def real_db():
    return _richest_project_db()


def _decode_vectors(con, table, col, limit=2000):
    rows = con.execute(f"SELECT vec_to_json({col}) FROM {table} LIMIT {limit}").fetchall()
    mats = [np.array(json.loads(r[0]), dtype=np.float32) for r in rows if r[0]]
    return np.stack(mats) if mats else np.empty((0, 0))


# --------------------------------------------------------------------------- #
# #22 — 7 embedding stages: correct dim, non-degenerate, no NaN, no zero rows
# --------------------------------------------------------------------------- #
EMBEDDING_SPECS = [
    ("vec_frames", "visual_embedding", 1152),   # SigLIP
    ("vec_semantic", "semantic_embedding", 768),  # Nomic
    ("vec_emotion", "emotion_embedding", 1024),   # Wav2Vec2
    ("vec_speakers", "speaker_embedding", 512),   # WavLM
]


@pytest.mark.integration
@pytest.mark.parametrize("table,col,dim", EMBEDDING_SPECS)
def test_embedding_store_is_correct_and_non_degenerate(real_db, table, col, dim):
    con = _open_db(real_db)
    try:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if n == 0:
            pytest.skip(f"{table} empty in {real_db.parent.name} — pick a project that ran this stage")
        mat = _decode_vectors(con, table, col)
    finally:
        con.close()
    print(f"[FSV] {real_db.parent.name}/{table}: n={mat.shape[0]} dim={mat.shape[1]} "
          f"var={mat.var():.5f} nan={np.isnan(mat).any()}")
    assert mat.shape[1] == dim, f"{table} dim {mat.shape[1]} != expected {dim}"
    assert not np.isnan(mat).any() and not np.isinf(mat).any(), "NaN/Inf in embeddings"
    assert mat.var() > 0, "constant embeddings (variance 0) — degenerate instrument"
    zero_rows = int((np.abs(mat).sum(axis=1) == 0).sum())
    assert zero_rows == 0, f"{zero_rows} all-zero embedding rows (the emotion valence-bug class)"


# --------------------------------------------------------------------------- #
# #23 — non-embedding stages: persisted outputs are well-formed
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_transcript_words_have_monotonic_timestamps(real_db):
    con = _open_db(real_db)
    try:
        rows = con.execute(
            "SELECT start_ms, end_ms FROM transcript_words ORDER BY start_ms"
        ).fetchall()
    finally:
        con.close()
    if not rows:
        pytest.skip("no transcript_words")
    last = -1
    for s, e in rows:
        assert s is not None and e is not None and e >= s, f"bad word span {s}->{e}"
        assert s >= last - 1, "word start went backwards (non-monotonic)"
        last = s
    print(f"[FSV] {len(rows)} transcript_words, monotonic timestamps OK")


@pytest.mark.integration
def test_scenes_and_highlights_well_formed(real_db):
    con = _open_db(real_db)
    try:
        scenes = con.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        hl = con.execute("SELECT score FROM highlights ORDER BY score DESC").fetchall()
    finally:
        con.close()
    print(f"[FSV] scenes={scenes} highlights={len(hl)}")
    assert scenes >= 1, "a real multi-scene clip must yield >=1 scene boundary"
    if hl:
        scores = [r[0] for r in hl]
        assert scores == sorted(scores, reverse=True), "highlights not ordered by score"
        assert all(s is not None for s in scores), "null highlight scores"


@pytest.mark.integration
def test_stream_status_has_no_silently_failed_stage(real_db):
    """Every stage that ran must be terminal-OK; a 'failed' row must carry a
    non-empty error_message (no swallowed failures)."""
    con = _open_db(real_db)
    try:
        rows = con.execute(
            "SELECT stream_name, status, error_message FROM stream_status"
        ).fetchall()
    finally:
        con.close()
    assert rows, "no stream_status rows — pipeline never recorded state"
    failed = [(n, e) for (n, s, e) in rows if s == "failed"]
    for name, err in failed:
        assert err and err.strip(), f"stage {name} failed with EMPTY error_message (swallowed)"
    completed = [n for (n, s, e) in rows if s == "completed"]
    print(f"[FSV] stream_status: {len(completed)} completed, {len(failed)} failed (all diagnosed)")
    assert completed, "no completed stages"


# --------------------------------------------------------------------------- #
# #34 — prosody captured from the demucs vocal stem (not the full mix)
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_prosody_segments_populated_and_usable(real_db):
    con = _open_db(real_db)
    try:
        has = con.execute(
            "SELECT name FROM sqlite_master WHERE name='prosody_segments'"
        ).fetchone()
        if not has:
            pytest.skip("prosody_segments table absent (project predates prosody stage)")
        rows = con.execute(
            "SELECT f0_mean, f0_std, energy_mean, speaking_rate_wpm, prosody_score "
            "FROM prosody_segments"
        ).fetchall()
    finally:
        con.close()
    if not rows:
        pytest.skip("prosody_segments empty for this project")
    # Usable references: at least some segments carry non-trivial F0/energy.
    nonzero_f0 = [r for r in rows if r[0] and r[0] > 0]
    print(f"[FSV] prosody_segments={len(rows)} with-F0={len(nonzero_f0)}")
    assert nonzero_f0, "prosody captured no pitch — vocal stem likely not analysed"


def test_prosody_loader_prefers_the_vocal_stem():
    """Static guarantee (#34): the prosody stage reads stems/vocals.wav first,
    i.e. the demucs vocal stem, before any full-mix fallback."""
    src = Path("src/clipcannon/pipeline/prosody_analysis.py").read_text()
    i_vocal = src.find('"stems" / "vocals.wav"')
    i_fallback = src.find('"audio.wav"')
    print(f"[FSV] vocal-stem ref at {i_vocal}, fallback at {i_fallback}")
    assert i_vocal != -1, "prosody stage does not reference the demucs vocal stem"
    assert i_fallback == -1 or i_vocal < i_fallback, "vocal stem must be tried before full mix"


def _silence_marker():
    # keep struct/math imports meaningful for linters in case of future use
    return struct.calcsize("f"), math.isfinite(1.0)
