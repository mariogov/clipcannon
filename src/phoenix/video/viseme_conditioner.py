"""Viseme conditioner for EchoMimicV3 integration.

Generates per-frame prompts and FLAME mouth parameters from text input
using the trained viseme lookup table.  This bridges the viseme system
with the video generation model.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

from .viseme_map import (
    MOUTH_PARAM_DIM,
    VISEME_PROMPTS,
    VisemeExtractor,
    phonemes_to_visemes,
    word_to_phonemes,
)


class VisemeConditioner:
    """Integrate viseme-based lip sync with EchoMimicV3 prompt conditioning.

    Takes a trained VisemeExtractor (or a pre-saved viseme table) and
    generates per-frame prompt augmentations and FLAME mouth parameter
    sequences for any input text.
    """

    def __init__(
        self,
        viseme_extractor: Optional[VisemeExtractor] = None,
        viseme_table_path: Optional[str | pathlib.Path] = None,
    ) -> None:
        """Initialize with a trained viseme table.

        Args:
            viseme_extractor: A VisemeExtractor with extracted table.
            viseme_table_path: Path to a saved .npz viseme table.
                Either extractor or path must be provided.
        """
        if viseme_extractor is not None:
            self._extractor = viseme_extractor
            self._table = viseme_extractor.viseme_table
        elif viseme_table_path is not None:
            self._extractor = None
            self._table = VisemeExtractor.load_table(viseme_table_path)
        else:
            raise ValueError(
                "Must provide either viseme_extractor or viseme_table_path"
            )

    @property
    def viseme_table(self) -> dict[str, np.ndarray]:
        return self._table

    def _build_timeline(
        self, text: str, duration_s: float
    ) -> list[tuple[str, float, float]]:
        """Build a viseme timeline from text.

        Returns list of (viseme_name, start_time_s, end_time_s).
        """
        words = text.split()
        if not words:
            return [("sil", 0.0, duration_s)]

        timeline: list[tuple[str, float, float]] = []
        word_dur = duration_s / len(words)

        for w_idx, word in enumerate(words):
            w_start = w_idx * word_dur
            w_end = w_start + word_dur

            phonemes = word_to_phonemes(word)
            if not phonemes:
                timeline.append(("sil", w_start, w_end))
                continue

            visemes = phonemes_to_visemes(phonemes)
            n = len(visemes)
            phone_dur = (w_end - w_start) / n

            for p_idx, vis in enumerate(visemes):
                p_start = w_start + p_idx * phone_dur
                p_end = p_start + phone_dur
                timeline.append((vis, p_start, p_end))

        return timeline

    def _viseme_at_time(
        self, timeline: list[tuple[str, float, float]], t: float
    ) -> str:
        """Find the active viseme at time t."""
        for vis, start, end in timeline:
            if start <= t < end:
                return vis
        return "sil"

    def condition_prompt_sequence(
        self,
        text: str,
        duration_s: float,
        base_prompt: str,
        fps: int = 25,
    ) -> list[tuple[int, str]]:
        """Generate per-frame prompts with viseme-specific mouth descriptions.

        For each frame:
          1. Determine which phoneme is active at this timestamp
          2. Look up the viseme for that phoneme
          3. Append the viseme mouth description to the base prompt

        Args:
            text: The spoken text.
            duration_s: Total duration in seconds.
            base_prompt: Base EchoMimicV3 prompt to augment.
            fps: Frame rate.

        Returns:
            List of (frame_idx, full_prompt_with_mouth_desc) tuples.
        """
        total_frames = int(duration_s * fps)
        if total_frames <= 0:
            return []

        timeline = self._build_timeline(text, duration_s)
        result: list[tuple[int, str]] = []

        for frame_idx in range(total_frames):
            t = frame_idx / fps
            vis = self._viseme_at_time(timeline, t)
            mouth_desc = VISEME_PROMPTS.get(vis, "mouth in neutral position")
            full_prompt = f"{base_prompt}, {mouth_desc}"
            result.append((frame_idx, full_prompt))

        return result

    def get_flame_params_for_text(
        self,
        text: str,
        duration_s: float,
        fps: int = 25,
    ) -> np.ndarray:
        """Get per-frame FLAME mouth parameters for the text.

        Returns:
            (N_frames, 13) array of mouth FLAME params where N_frames =
            int(duration_s * fps).  Each row is the viseme's average
            mouth parameters from Santa's training data.
        """
        total_frames = int(duration_s * fps)
        if total_frames <= 0:
            return np.zeros((0, MOUTH_PARAM_DIM), dtype=np.float32)

        timeline = self._build_timeline(text, duration_s)
        default_params = np.zeros(MOUTH_PARAM_DIM, dtype=np.float32)

        params = np.zeros((total_frames, MOUTH_PARAM_DIM), dtype=np.float32)
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            vis = self._viseme_at_time(timeline, t)
            params[frame_idx] = self._table.get(vis, default_params)

        return params

    def get_viseme_sequence(
        self,
        text: str,
        duration_s: float,
        fps: int = 25,
    ) -> list[tuple[int, str]]:
        """Get per-frame viseme names (without FLAME params).

        Returns:
            List of (frame_idx, viseme_name) tuples.
        """
        total_frames = int(duration_s * fps)
        if total_frames <= 0:
            return []

        timeline = self._build_timeline(text, duration_s)
        result: list[tuple[int, str]] = []

        for frame_idx in range(total_frames):
            t = frame_idx / fps
            vis = self._viseme_at_time(timeline, t)
            result.append((frame_idx, vis))

        return result

    def interpolate_flame_params(
        self,
        text: str,
        duration_s: float,
        fps: int = 25,
        smoothing_window: int = 3,
    ) -> np.ndarray:
        """Get smoothed per-frame FLAME params with interpolation.

        Applies a moving average to smooth transitions between visemes,
        producing more natural mouth movement.

        Args:
            text: Spoken text.
            duration_s: Total duration.
            fps: Frame rate.
            smoothing_window: Size of the moving average window (must be odd).

        Returns:
            (N_frames, 13) smoothed FLAME mouth params.
        """
        raw = self.get_flame_params_for_text(text, duration_s, fps)
        if raw.shape[0] <= 1:
            return raw

        # Ensure odd window
        w = max(1, smoothing_window)
        if w % 2 == 0:
            w += 1

        # Uniform moving average along the time axis
        pad = w // 2
        padded = np.pad(raw, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.zeros_like(raw)
        for i in range(raw.shape[0]):
            smoothed[i] = padded[i : i + w].mean(axis=0)

        return smoothed
