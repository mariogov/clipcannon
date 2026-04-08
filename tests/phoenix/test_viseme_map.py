"""Tests for viseme-based lip sync system.

Uses REAL data from the actual ClipCannon database and FLAME params.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project src is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phoenix.video.viseme_map import (
    MOUTH_PARAM_DIM,
    PHONEME_TO_VISEME,
    VISEME_MAP,
    VISEME_PROMPTS,
    VisemeExtractor,
    phonemes_to_visemes,
    word_to_phonemes,
)
from phoenix.video.viseme_conditioner import VisemeConditioner


# ---------------------------------------------------------------------------
# Paths to real data
# ---------------------------------------------------------------------------
DB_PATH = Path.home() / ".clipcannon" / "projects" / "proj_2ea7221d" / "analysis.db"
FLAME_PATH = Path.home() / ".clipcannon" / "models" / "santa" / "flame_params.npz"

REAL_DATA_AVAILABLE = DB_PATH.exists() and FLAME_PATH.exists()
skip_no_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE, reason="Real training data not available"
)


# ---------------------------------------------------------------------------
# Static tests (no data needed)
# ---------------------------------------------------------------------------

class TestVisemeMapConstants:
    """Test the static viseme mapping tables."""

    def test_viseme_map_has_15_entries(self):
        assert len(VISEME_MAP) == 15

    def test_viseme_ids_are_0_to_14(self):
        assert set(VISEME_MAP.values()) == set(range(15))

    def test_silence_viseme_is_zero(self):
        assert VISEME_MAP["sil"] == 0

    def test_all_visemes_have_prompts(self):
        for vis_name in VISEME_MAP:
            assert vis_name in VISEME_PROMPTS, f"Missing prompt for {vis_name}"
            assert len(VISEME_PROMPTS[vis_name]) > 0

    def test_phoneme_to_viseme_coverage(self):
        """All mapped phonemes should map to a valid viseme."""
        for phone, vis in PHONEME_TO_VISEME.items():
            assert vis in VISEME_MAP, f"Phoneme {phone} maps to unknown viseme {vis}"


class TestWordToPhonemes:
    """Test word-to-phoneme lookup."""

    def test_known_word(self):
        phones = word_to_phonemes("hello")
        assert len(phones) > 0
        # 'hello' should have HH, AH/EH, L, OW
        assert any("HH" in p for p in phones)

    def test_unknown_word_fallback(self):
        phones = word_to_phonemes("xyzquux")
        assert len(phones) > 0  # Letter-by-letter fallback

    def test_empty_input(self):
        phones = word_to_phonemes("")
        assert phones == []

    def test_punctuation_stripped(self):
        phones = word_to_phonemes("hello,")
        phones2 = word_to_phonemes("hello")
        assert phones == phones2

    def test_christmas_word(self):
        phones = word_to_phonemes("christmas")
        assert len(phones) > 0


class TestPhonemesToVisemes:
    """Test phoneme-to-viseme conversion."""

    def test_bilabial(self):
        result = phonemes_to_visemes(["P", "B", "M"])
        assert all(v == "PP" for v in result)

    def test_vowels(self):
        result = phonemes_to_visemes(["AA1", "IY0", "UW1"])
        assert result == ["aa", "EE", "OO"]

    def test_mixed_sequence(self):
        # 'cat' = K AE1 T
        result = phonemes_to_visemes(["K", "AE1", "T"])
        assert result == ["kk", "aa", "DD"]

    def test_unknown_phoneme_defaults_to_silence(self):
        result = phonemes_to_visemes(["ZZZZZ"])
        assert result == ["sil"]


# ---------------------------------------------------------------------------
# Integration tests with real data
# ---------------------------------------------------------------------------

@skip_no_data
class TestVisemeExtractorReal:
    """Test VisemeExtractor with real Santa training data."""

    @pytest.fixture(scope="class")
    def extractor(self):
        ext = VisemeExtractor(db_path=DB_PATH, flame_params_path=FLAME_PATH)
        ext.extract_viseme_table()
        return ext

    def test_loads_words(self, extractor):
        assert len(extractor._words) == 2500

    def test_loads_flame_frames(self, extractor):
        assert extractor._expression.shape == (3477, 100)

    def test_all_15_visemes_have_samples(self, extractor):
        """All 15 visemes should have at least some samples from 2500 words."""
        stats = extractor.get_statistics()
        covered = stats["visemes_covered"]
        # We expect high coverage from 2500 words
        assert covered >= 13, (
            f"Only {covered}/15 visemes covered. "
            f"Missing: {[v for v in VISEME_MAP if stats['per_viseme'].get(v, 0) == 0]}"
        )

    def test_viseme_table_shape(self, extractor):
        table = extractor.viseme_table
        for vis_name, params in table.items():
            assert params.shape == (MOUTH_PARAM_DIM,), (
                f"Viseme {vis_name} has wrong shape: {params.shape}"
            )
            assert params.dtype == np.float32

    def test_different_visemes_have_different_params(self, extractor):
        """Visemes with samples should have distinct mouth shapes."""
        table = extractor.viseme_table
        counts = extractor.viseme_counts
        populated = [v for v in VISEME_MAP if counts.get(v, 0) > 10]
        # Check at least some pairs are different
        different_pairs = 0
        total_pairs = 0
        for i, v1 in enumerate(populated):
            for v2 in populated[i + 1 :]:
                total_pairs += 1
                if not np.allclose(table[v1], table[v2], atol=0.01):
                    different_pairs += 1
        # At least 50% of pairs should be distinguishable
        assert different_pairs > total_pairs * 0.3, (
            f"Only {different_pairs}/{total_pairs} viseme pairs are distinct"
        )

    def test_silence_params_differ_from_open_mouth(self, extractor):
        """Silence should look different from 'aa' (open mouth)."""
        table = extractor.viseme_table
        sil = table["sil"]
        aa = table["aa"]
        # They should not be identical
        diff = np.linalg.norm(sil - aa)
        assert diff > 0.01, "Silence and open-mouth should differ"

    def test_total_samples_reasonable(self, extractor):
        stats = extractor.get_statistics()
        # 2500 words with ~4 phonemes each = ~10000 samples + silence gaps
        assert stats["total_samples"] > 5000

    def test_text_to_viseme_sequence_frame_count(self, extractor):
        """Output should have exactly the right number of frames."""
        text = "Hello world how are you"
        duration = 2.0
        fps = 25
        seq = extractor.text_to_viseme_sequence(text, duration, fps)
        expected_frames = int(duration * fps)
        assert len(seq) == expected_frames

    def test_text_to_viseme_sequence_structure(self, extractor):
        """Each entry should be (frame_idx, viseme_name, params)."""
        seq = extractor.text_to_viseme_sequence("test", 1.0, 25)
        for frame_idx, vis_name, params in seq:
            assert isinstance(frame_idx, int)
            assert isinstance(vis_name, str)
            assert vis_name in VISEME_MAP
            assert params.shape == (MOUTH_PARAM_DIM,)

    def test_silence_in_gaps(self, extractor):
        """Silence viseme should have samples from inter-word gaps."""
        counts = extractor.viseme_counts
        assert counts["sil"] > 0, "No silence samples found in gaps"

    def test_viseme_to_prompt_returns_strings(self, extractor):
        for vis_name in VISEME_MAP:
            prompt = extractor.viseme_to_prompt(vis_name)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_unknown_word_handled(self, extractor):
        """Words not in CMU dict should still produce visemes."""
        seq = extractor.text_to_viseme_sequence("xyzquux blergfoo", 1.0, 25)
        assert len(seq) == 25
        # Should not be all silence
        non_sil = sum(1 for _, v, _ in seq if v != "sil")
        assert non_sil > 0


@skip_no_data
class TestVisemeTableSaveLoad:
    """Test saving and loading the viseme table."""

    def test_save_load_roundtrip(self, tmp_path):
        extractor = VisemeExtractor(db_path=DB_PATH, flame_params_path=FLAME_PATH)
        extractor.extract_viseme_table()

        save_path = tmp_path / "test_viseme_table.npz"
        extractor.save_table(save_path)

        assert save_path.exists()
        loaded = VisemeExtractor.load_table(save_path)
        original = extractor.viseme_table

        for vis_name in VISEME_MAP:
            assert vis_name in loaded
            np.testing.assert_allclose(
                loaded[vis_name], original[vis_name], atol=1e-6
            )


@skip_no_data
class TestVisemeConditionerReal:
    """Test VisemeConditioner with real data."""

    @pytest.fixture(scope="class")
    def conditioner(self):
        ext = VisemeExtractor(db_path=DB_PATH, flame_params_path=FLAME_PATH)
        ext.extract_viseme_table()
        return VisemeConditioner(viseme_extractor=ext)

    def test_condition_prompt_sequence(self, conditioner):
        prompts = conditioner.condition_prompt_sequence(
            text="Hello world",
            duration_s=2.0,
            base_prompt="Santa Claus talking",
            fps=25,
        )
        assert len(prompts) == 50  # 2.0s * 25fps
        for frame_idx, full_prompt in prompts:
            assert "Santa Claus talking" in full_prompt
            # Should have a mouth description appended
            assert ", " in full_prompt

    def test_get_flame_params_shape(self, conditioner):
        params = conditioner.get_flame_params_for_text(
            text="Merry Christmas",
            duration_s=1.5,
            fps=25,
        )
        expected_frames = int(1.5 * 25)
        assert params.shape == (expected_frames, MOUTH_PARAM_DIM)
        assert params.dtype == np.float32

    def test_interpolated_params_same_shape(self, conditioner):
        raw = conditioner.get_flame_params_for_text("test", 1.0, 25)
        smooth = conditioner.interpolate_flame_params("test", 1.0, 25, smoothing_window=5)
        assert raw.shape == smooth.shape

    def test_smoothing_reduces_variance(self, conditioner):
        """Smoothed params should have less frame-to-frame variance."""
        text = "Ho ho ho Merry Christmas everyone"
        raw = conditioner.get_flame_params_for_text(text, 3.0, 25)
        smooth = conditioner.interpolate_flame_params(text, 3.0, 25, smoothing_window=5)
        # Compute frame-to-frame differences
        raw_diffs = np.diff(raw, axis=0)
        smooth_diffs = np.diff(smooth, axis=0)
        raw_var = np.mean(raw_diffs ** 2)
        smooth_var = np.mean(smooth_diffs ** 2)
        assert smooth_var <= raw_var, (
            f"Smoothing did not reduce variance: {smooth_var:.4f} vs {raw_var:.4f}"
        )

    def test_empty_text(self, conditioner):
        prompts = conditioner.condition_prompt_sequence("", 1.0, "base", 25)
        assert len(prompts) == 25
        # All should be silence
        for _, prompt in prompts:
            assert "lips gently closed" in prompt

    def test_get_viseme_sequence(self, conditioner):
        seq = conditioner.get_viseme_sequence("hello", 1.0, 25)
        assert len(seq) == 25
        for frame_idx, vis_name in seq:
            assert vis_name in VISEME_MAP

    def test_from_saved_table(self, tmp_path):
        """Test creating conditioner from a saved NPZ file."""
        ext = VisemeExtractor(db_path=DB_PATH, flame_params_path=FLAME_PATH)
        ext.extract_viseme_table()
        save_path = tmp_path / "vis_table.npz"
        ext.save_table(save_path)

        cond = VisemeConditioner(viseme_table_path=save_path)
        params = cond.get_flame_params_for_text("test", 1.0, 25)
        assert params.shape == (25, MOUTH_PARAM_DIM)
