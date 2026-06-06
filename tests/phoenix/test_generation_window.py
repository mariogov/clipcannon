"""Unit tests for the generation-window guard (deterministic, no model needed).

Synthetic known inputs with known expected outputs, per the EchoMimicV3
sliding-window policy (partial=113, overlap=8, VAE tcr=4).
"""

import pytest

from phoenix.video.generation_window import (
    MODEL_MAX_FRAMES,
    Chunk,
    WindowExceeded,
    align_partial,
    frames_for_seconds,
    plan_chunks,
    validate_window,
)


def test_align_partial_known_values():
    # ((p-1)//4)*4 + 1
    assert align_partial(113) == 113   # (112//4)*4+1 = 113
    assert align_partial(20) == 17     # (19//4)*4+1  = 17
    assert align_partial(25) == 25     # (24//4)*4+1  = 25
    assert align_partial(1) == 1
    assert align_partial(4) == 1


def test_frames_for_seconds():
    assert frames_for_seconds(5.0, 25) == 125
    assert frames_for_seconds(1.0, 25) == 25


def test_validate_window_guard():
    validate_window(113)               # ok, at the limit
    with pytest.raises(WindowExceeded):
        validate_window(200)           # the Hunyuan-style overrun
    with pytest.raises(WindowExceeded):
        validate_window(250)


def test_single_chunk_for_short_clip():
    # 1s @ 25fps = 25 frames < 113 -> exactly one chunk
    chunks = plan_chunks(25)
    assert chunks == [Chunk(0, 0, 25)]


def test_invariant_no_chunk_exceeds_window_5s():
    # 5s @ 25fps = 125 frames -> multiple chunks, none over the safe window
    chunks = plan_chunks(125)
    assert len(chunks) >= 2
    assert chunks[0].length == MODEL_MAX_FRAMES          # first chunk is full 113
    assert all(c.length <= MODEL_MAX_FRAMES for c in chunks)  # the guard property
    assert all(c.start >= 0 for c in chunks)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)                      # monotonic, non-overlapping starts


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        plan_chunks(0)                                   # empty
    with pytest.raises(ValueError):
        plan_chunks(100, overlap_video_length=200)       # overlap >= partial


def test_single_frame():
    assert plan_chunks(1) == [Chunk(0, 0, 1)]
