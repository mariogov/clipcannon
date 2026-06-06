"""Generation-window planning + identity-drift guard for EchoMimicV3.

Root-cause hardening for the identity-drift failure in
``docs/inspect/04_output_quality.md``: HunyuanVideo-Avatar generated 250 frames
in one shot, far beyond its ~129-frame training window, and the subject's
identity diverged (NCC 0.87 -> -0.15) by the 5s mark.

EchoMimicV3 avoids this with sliding-window chunked generation: each chunk is at
most ``partial_video_length`` frames (default 113, aligned to the VAE temporal
compression ratio), and consecutive chunks share ``overlap_video_length`` frames
that are linearly blended and re-anchored to the reference image. This module
makes that policy explicit and *guards* it:

  * ``align_partial`` reproduces infer_preview's frame alignment
    ``((p-1)//tcr)*tcr + 1``.
  * ``plan_chunks`` returns the exact (start, length) windows infer_preview will
    generate for a given total frame count — so the chunking is auditable.
  * ``validate_window`` raises if a single chunk would exceed the model's safe
    window (the condition that caused the Hunyuan drift). No silent clamping.

This is deterministic arithmetic with no model dependency, so it is unit-tested
against synthetic known inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

# EchoMimicV3-preview validated per-chunk window (infer_preview partial default).
MODEL_MAX_FRAMES = 113
# Wan VAE temporal compression ratio (frames -> latent frames).
TEMPORAL_COMPRESSION_RATIO = 4


class WindowExceeded(ValueError):
    """Raised when a requested per-chunk window exceeds the model's safe max."""


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int          # first frame (inclusive)
    length: int         # frames generated in this chunk (aligned)


def align_partial(partial: int, tcr: int = TEMPORAL_COMPRESSION_RATIO) -> int:
    """Align a partial length to the VAE temporal grid, as infer_preview does.

    ``((partial - 1) // tcr) * tcr + 1``. A 1-frame request stays 1.
    """
    if partial <= 1:
        return 1
    return ((partial - 1) // tcr) * tcr + 1


def validate_window(
    partial_video_length: int,
    model_max_frames: int = MODEL_MAX_FRAMES,
) -> None:
    """Raise WindowExceeded if a single chunk would exceed the safe window.

    This is the guard against the Hunyuan-style one-shot-overrun drift.
    """
    if partial_video_length > model_max_frames:
        raise WindowExceeded(
            f"per-chunk window {partial_video_length} > model max "
            f"{model_max_frames} frames; would cause identity drift. "
            f"Use chunked generation (plan_chunks) instead of one-shot."
        )


def plan_chunks(
    total_frames: int,
    partial_video_length: int = MODEL_MAX_FRAMES,
    overlap_video_length: int = 8,
    tcr: int = TEMPORAL_COMPRESSION_RATIO,
    model_max_frames: int = MODEL_MAX_FRAMES,
) -> list[Chunk]:
    """Plan the sliding-window chunks infer_preview will generate.

    Mirrors infer_preview's loop: each chunk advances by
    ``partial - overlap`` frames; the final chunk is shrunk to the remainder
    and re-aligned. Raises on invalid inputs (fail loud, no silent fallback).
    """
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")
    if overlap_video_length < 0 or overlap_video_length >= partial_video_length:
        raise ValueError(
            f"overlap {overlap_video_length} must be in [0, {partial_video_length})"
        )
    base_partial = align_partial(partial_video_length, tcr)
    validate_window(base_partial, model_max_frames)

    if total_frames == 1:
        return [Chunk(0, 0, 1)]

    chunks: list[Chunk] = []
    init = 0
    idx = 0
    advance = base_partial - overlap_video_length
    while init < total_frames:
        remaining = total_frames - init
        length = base_partial if remaining >= base_partial else align_partial(remaining, tcr)
        if length <= 0:
            break
        chunks.append(Chunk(idx, init, length))
        if init + length >= total_frames:
            break
        init += advance
        idx += 1
    return chunks


def frames_for_seconds(seconds: float, fps: int = 25) -> int:
    """Frames for a given duration (infer_preview: int(duration * fps))."""
    return int(seconds * fps)
