"""Self-FSV for the FSV harness (GitHub issue #57).

The harness's own source of truth is a REAL sqlite DB and REAL media files
(generated here with ffmpeg). We prove each primitive reports the actual stored
state and RAISES on the degenerate/missing cases — so a future test built on it
cannot silently pass while the system is broken.
"""
from __future__ import annotations

import struct
import subprocess

import numpy as np
import pytest

from clipcannon.db.connection import get_connection
from clipcannon.db.schema import create_project_db
from tests import fsv_harness as H

PROJECT_ID = "proj_fsv_harness"


def _seed(db_path, valences):
    """Seed a real project + emotion_curve + vec_emotion with known data."""
    conn = get_connection(db_path, enable_vec=True, dict_rows=False)
    try:
        conn.execute(
            "INSERT INTO project (project_id, name, source_path, source_sha256, "
            "duration_ms, resolution, fps, codec) VALUES (?,?,?,?,?,?,?,?)",
            (PROJECT_ID, "h", "/tmp/h.mp4", "0" * 64, 60000, "1920x1080", 30.0, "h264"),
        )
        for i, val in enumerate(valences):
            conn.execute(
                "INSERT INTO emotion_curve (project_id, start_ms, end_ms, arousal, valence, energy) "
                "VALUES (?,?,?,?,?,?)",
                (PROJECT_ID, i * 2500, i * 2500 + 5000, 0.2 + 0.1 * i, val, 0.1),
            )
            rng = np.random.default_rng(i + 1)
            emb = rng.standard_normal(1024).astype(np.float32)
            conn.execute(
                "INSERT INTO vec_emotion (project_id, start_ms, end_ms, emotion_embedding) "
                "VALUES (?,?,?,?)",
                (PROJECT_ID, i * 2500, i * 2500 + 5000, struct.pack("<1024f", *emb.tolist())),
            )
        conn.commit()
    finally:
        conn.close()


def test_rowcount_reads_source_of_truth(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed(db, [0.8, 0.2, 0.55])
    assert H.assert_rowcount(db, "emotion_curve", 3) == 3
    with pytest.raises(AssertionError, match="expected 99"):
        H.assert_rowcount(db, "emotion_curve", 99)


def test_vector_store_happy(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed(db, [0.8, 0.2, 0.55])
    arr = H.assert_vector_store(db, "vec_emotion", "emotion_embedding", 1024, min_count=3)
    assert arr.shape == (3, 1024)
    assert np.any(arr)


def test_vector_store_wrong_dim_raises(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed(db, [0.8, 0.2, 0.55])
    with pytest.raises(AssertionError, match="dim"):
        H.read_vectors(db, "vec_emotion", "emotion_embedding", 512)  # stored is 1024


def test_vector_store_all_zero_raises(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    # seed three all-zero embeddings (the dead-instrument signature)
    conn = get_connection(db, enable_vec=True, dict_rows=False)
    try:
        conn.execute(
            "INSERT INTO project (project_id, name, source_path, source_sha256, "
            "duration_ms, resolution, fps, codec) VALUES (?,?,?,?,?,?,?,?)",
            (PROJECT_ID, "h", "/tmp/h.mp4", "0" * 64, 1, "1x1", 1.0, "h264"),
        )
        zero = struct.pack("<1024f", *([0.0] * 1024))
        for i in range(3):
            conn.execute(
                "INSERT INTO vec_emotion (project_id, start_ms, end_ms, emotion_embedding) "
                "VALUES (?,?,?,?)",
                (PROJECT_ID, i, i + 1, zero),
            )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(AssertionError, match="ALL vectors are zero"):
        H.assert_vector_store(db, "vec_emotion", "emotion_embedding", 1024, min_count=3)


def test_no_nan_raises():
    bad = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    with pytest.raises(AssertionError, match="non-finite"):
        H.assert_no_nan(bad, "probe")


def test_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        H.table_rowcount(tmp_path / "nope.db", "emotion_curve")


def test_snapshot_diff_proves_trigger(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    before = H.snapshot(db, ["emotion_curve"])
    _seed(db, [0.8, 0.2, 0.55])
    after = H.snapshot(db, ["emotion_curve"])
    delta = H.diff_snapshot(before, after)
    assert delta["emotion_curve"] == 3


def _have_ffmpeg() -> bool:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_media_video_via_ffmpeg(tmp_path):
    out = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
         "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, check=True,
    )
    H.assert_media(out, width=320, height=240, min_duration_s=1.9, has_audio=False)


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_media_audio_via_ffmpeg(tmp_path):
    out = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=16000", str(out)],
        capture_output=True, check=True,
    )
    H.assert_media(out, has_audio=True, sample_rate=16000, min_duration_s=1.9)


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_media_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(AssertionError, match="empty"):
        H.ffprobe_media(empty)
