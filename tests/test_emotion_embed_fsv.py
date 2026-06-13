"""Full State Verification for the emotion instrument (GitHub issue #18).

These tests prove the emotion axis is a GROUNDED, non-degenerate instrument:

1. Persistence FSV — write synthetic-but-known emotion results into a REAL
   project database (real schema, real sqlite-vec), then independently read
   `emotion_curve` and `vec_emotion` back to prove the exact values landed.
   This is Full State Verification: we do not trust return values, we inspect
   the source of truth (the DB) after the write.

2. Differentiation-contract guard — the degenerate signatures that previously
   shipped (all-zero embeddings; constant valence) MUST raise, and varied real
   signal MUST pass.

3. Grounded-model FSV — runs the actual dimensional-SER model on real audio and
   proves valence/arousal vary and the embedding is non-zero. Requires torch +
   transformers + the pre-cached model + audio; if any is absent the test SKIPS
   with the exact missing dependency (a skip, never a false pass).

No mocks. No fallbacks.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from clipcannon.db.connection import get_connection
from clipcannon.db.schema import create_project_db
from clipcannon.exceptions import PipelineError
from clipcannon.pipeline.emotion_embed import (
    EMBEDDING_DIM,
    _assert_not_degenerate,
    _insert_results,
)

PROJECT_ID = "proj_fsv_emotion"


def _seed_project(db_path) -> None:
    """Insert the parent project row required by emotion_curve's FK."""
    conn = get_connection(db_path, enable_vec=False, dict_rows=False)
    try:
        conn.execute(
            "INSERT INTO project (project_id, name, source_path, source_sha256, "
            "duration_ms, resolution, fps, codec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (PROJECT_ID, "fsv", "/tmp/fsv.mp4", "0" * 64, 60000, "1920x1080", 30.0, "h264"),
        )
        conn.commit()
    finally:
        conn.close()


def _synthetic_results(valences: list[float]) -> list[dict[str, object]]:
    """Build emotion results with KNOWN values and distinct real embeddings.

    Each window gets a deterministic, non-degenerate 1024-d embedding so we can
    read it back byte-for-byte from vec_emotion.
    """
    results: list[dict[str, object]] = []
    for i, val in enumerate(valences):
        rng = np.random.default_rng(i + 1)
        emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        results.append(
            {
                "start_ms": i * 2500,
                "end_ms": i * 2500 + 5000,
                "energy": round(0.1 + 0.05 * i, 4),
                "arousal": round(0.2 + 0.1 * i, 4),
                "valence": round(val, 4),
                "embedding": emb,
            }
        )
    return results


def test_insert_results_persists_to_real_db(tmp_path):
    """FSV: the source of truth (emotion_curve + vec_emotion) holds the data."""
    db_path = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db_path)

    # Known input: 3 windows with clearly different valences.
    valences = [0.80, 0.20, 0.55]
    results = _synthetic_results(valences)

    counts = _insert_results(db_path, PROJECT_ID, results)
    assert counts["emotion_curve"] == 3
    assert counts["vec_emotion"] == 3

    # --- INSPECT THE SOURCE OF TRUTH with a fresh, independent connection ---
    conn = get_connection(db_path, enable_vec=True, dict_rows=False)
    try:
        rows = conn.execute(
            "SELECT start_ms, end_ms, arousal, valence, energy "
            "FROM emotion_curve ORDER BY start_ms"
        ).fetchall()
        # Evidence of success: the actual data residing in the system.
        print("\n[FSV] emotion_curve contents:")
        for r in rows:
            print(f"   start={r[0]} end={r[1]} arousal={r[2]} valence={r[3]} energy={r[4]}")

        assert len(rows) == 3
        db_valences = [round(r[3], 4) for r in rows]
        assert db_valences == valences, f"valence mismatch: {db_valences} != {valences}"

        # The bug we are fixing: valence must NOT be a constant 0.5.
        assert len(set(db_valences)) == 3, "valence collapsed to a constant"
        assert 0.5 not in db_valences or db_valences.count(0.5) <= 1

        # Vector store: count + dim + non-zero + byte-exact round trip.
        vrows = conn.execute(
            "SELECT start_ms, emotion_embedding FROM vec_emotion ORDER BY start_ms"
        ).fetchall()
        assert len(vrows) == 3
        for i, (_start, blob) in enumerate(vrows):
            vec = np.array(struct.unpack(f"{EMBEDDING_DIM}f", blob), dtype=np.float32)
            assert vec.shape[0] == EMBEDDING_DIM
            assert np.any(vec), f"window {i} embedding is all-zero (dead instrument)"
            np.testing.assert_allclose(vec, results[i]["embedding"], rtol=1e-5, atol=1e-6)
        print(f"[FSV] vec_emotion: {len(vrows)} rows, dim={EMBEDDING_DIM}, all non-zero ✓")
    finally:
        conn.close()


def test_insert_rejects_missing_embedding(tmp_path):
    """No silent partial writes: a non-ndarray embedding must raise."""
    db_path = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db_path)
    results = _synthetic_results([0.8, 0.2])
    results[1]["embedding"] = None  # corrupt one window
    with pytest.raises(PipelineError, match="embedding"):
        _insert_results(db_path, PROJECT_ID, results)


def test_degeneracy_guard_rejects_zero_embeddings():
    """Edge case: all-zero embeddings (the old fallback) → must raise."""
    results = _synthetic_results([0.8, 0.2, 0.5])
    for r in results:
        r["embedding"] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    with pytest.raises(PipelineError, match="all zero"):
        _assert_not_degenerate(results)


def test_degeneracy_guard_rejects_constant_valence():
    """Edge case: constant valence across ≥3 windows → must raise."""
    results = _synthetic_results([0.5, 0.5, 0.5])
    with pytest.raises(PipelineError, match="constant"):
        _assert_not_degenerate(results)


def test_degeneracy_guard_passes_varied_signal():
    """Happy path: varied valence + real embeddings → no raise."""
    results = _synthetic_results([0.8, 0.2, 0.55])
    _assert_not_degenerate(results)  # must not raise


def test_degeneracy_guard_allows_two_equal_windows():
    """A 2-window clip with equal valence is not enough to call it degenerate."""
    results = _synthetic_results([0.5, 0.5])
    _assert_not_degenerate(results)  # must not raise (guard needs ≥3)


@pytest.mark.integration
def test_grounded_model_produces_varied_valence(tmp_path):
    """GPU-host FSV: the real SER model yields grounded, varied valence.

    Skips (never false-passes) if torch/transformers/model/audio are absent.
    """
    torch = pytest.importorskip("torch", reason="torch required for real SER FSV")
    pytest.importorskip("transformers", reason="transformers required for real SER FSV")
    from clipcannon.pipeline.emotion_embed import MODEL_ID, _compute_emotion_model

    try:
        from transformers import Wav2Vec2Processor  # noqa: F401

        _ = _build_probe(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SER model '{MODEL_ID}' not pre-cached: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = 16000
    # Two acoustically distinct 5s windows (real synthetic audio): a calm low
    # tone vs an agitated noisy sweep. We assert the model separates them — we
    # do not assert exact values (that would be brittle), only that the
    # instrument carries information (non-constant valence, non-zero embedding).
    rng = np.random.default_rng(0)
    t = np.linspace(0, 5, sr * 5, dtype=np.float32)
    calm = 0.05 * np.sin(2 * np.pi * 110 * t).astype(np.float32)
    agitated = (0.4 * np.sin(2 * np.pi * 660 * t) + 0.3 * rng.standard_normal(t.shape[0])).astype(np.float32)
    segments = [(0, 5000, calm), (2500, 7500, agitated)]

    results = _compute_emotion_model(segments, device)
    valences = [r["valence"] for r in results]
    print(f"\n[FSV] real-model valences: {valences}")
    assert len(set(valences)) > 1, "model produced constant valence — ungrounded"
    for r in results:
        assert np.any(r["embedding"]), "real embedding is all-zero"
        assert 0.0 <= r["valence"] <= 1.0
        assert 0.0 <= r["arousal"] <= 1.0


def _build_probe(model_id: str):
    """Confirm the model is locally cached without downloading."""
    from transformers import Wav2Vec2Processor

    return Wav2Vec2Processor.from_pretrained(model_id, local_files_only=True)


@pytest.mark.integration
def test_grounded_model_on_real_speech_writes_to_db(tmp_path):
    """End-to-end GPU FSV: real speech -> SER model -> emotion_curve + vec_emotion.

    Trigger: a real WAV of human speech. Process: dimensional-SER on the 5090.
    Outcome: per-window valence/arousal/energy rows + 1024-d embeddings PERSISTED
    and read back from the real DB. Proves the instrument is alive and grounded
    on real data (not just synthetic tones).
    """
    import glob

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from clipcannon.pipeline.emotion_embed import (
        MODEL_ID,
        _compute_emotion_model,
        _load_audio,
        _segment_audio,
    )

    wavs = sorted(glob.glob(str(Path.home() / ".clipcannon/voice_data/*/wavs/*.wav")))
    wavs = [w for w in wavs if "_trimmed" not in w]
    if not wavs:
        pytest.skip("no real speech wavs under ~/.clipcannon/voice_data")
    try:
        _build_probe(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SER model '{MODEL_ID}' not pre-cached: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Concatenate a few real clips so we get >=3 windows of real speech.
    import numpy as np

    chunks = []
    for w in wavs[:4]:
        audio, sr = _load_audio(Path(w))
        chunks.append(audio)
    audio = np.concatenate(chunks)
    segments = _segment_audio(audio, 16000)
    assert len(segments) >= 3, f"need >=3 windows, got {len(segments)}"

    results = _compute_emotion_model(segments, device)

    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db)
    counts = _insert_results(db, PROJECT_ID, results)

    # --- INSPECT THE SOURCE OF TRUTH ---
    conn = get_connection(db, enable_vec=True, dict_rows=False)
    try:
        rows = conn.execute(
            "SELECT start_ms, arousal, valence, energy FROM emotion_curve ORDER BY start_ms"
        ).fetchall()
        print("\n[FSV] REAL SPEECH emotion_curve (from the 5090):")
        for r in rows:
            print(f"   t={r[0]}ms arousal={r[1]} valence={r[2]} energy={r[3]}")
        vrows = conn.execute("SELECT emotion_embedding FROM vec_emotion").fetchall()
    finally:
        conn.close()

    assert counts["emotion_curve"] == len(results) == counts["vec_emotion"]
    valences = [r[2] for r in rows]
    # Grounded signal on real speech: valence varies window-to-window and sits in range.
    assert len(set(valences)) > 1, f"valence constant on real speech: {valences}"
    assert all(0.0 <= v <= 1.0 for v in valences)
    # Every embedding is a real, non-zero 1024-d vector.
    for blob in vrows:
        vec = np.array(struct.unpack("<1024f", blob[0]), dtype=np.float32)
        assert vec.shape[0] == 1024 and np.any(vec)
    print(f"[FSV] real-speech valence range: {min(valences)}..{max(valences)} across {len(valences)} windows ✓")
