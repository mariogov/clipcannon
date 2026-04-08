"""Viseme-based lip sync system.

Extracts mouth shapes from Santa's training video and maps them to
MPEG-4 standard visemes for precise per-phoneme lip sync.

Training pipeline:
    Words (2500) -> Phonemes (CMU dict) -> Visemes (15 groups)
    FLAME expressions at each word timestamp -> mouth params
    Build: viseme -> average FLAME mouth params lookup table

Inference pipeline:
    New text -> phonemes -> visemes -> FLAME mouth params per frame
    -> embed in prompt conditioning for EchoMimicV3
"""

from __future__ import annotations

import pathlib
import re
import sqlite3
from typing import Optional

import numpy as np
from nltk.corpus import cmudict


# ---------------------------------------------------------------------------
# 15 standard MPEG-4 visemes
# ---------------------------------------------------------------------------

VISEME_MAP: dict[str, int] = {
    "sil": 0,     # Silence -- mouth closed
    "PP": 1,      # P, B, M -- bilabial
    "FF": 2,      # F, V -- labiodental
    "TH": 3,      # TH, DH -- dental
    "DD": 4,      # T, D -- alveolar stop
    "kk": 5,      # K, G, NG -- velar
    "CH": 6,      # SH, CH, JH, ZH -- postalveolar
    "SS": 7,      # S, Z -- sibilant
    "nn": 8,      # N, L -- nasal/liquid
    "RR": 9,      # R, ER -- rhotic
    "aa": 10,     # AA, AH, AE -- open
    "EE": 11,     # IY, EY -- front spread
    "ih": 12,     # IH, EH -- mid front
    "OH": 13,     # AO, OW -- back rounded
    "OO": 14,     # UW, UH, W -- close rounded
}

# Reverse: viseme index -> name
VISEME_NAMES: dict[int, str] = {v: k for k, v in VISEME_MAP.items()}

# ---------------------------------------------------------------------------
# CMU phoneme -> viseme mapping
# ---------------------------------------------------------------------------

PHONEME_TO_VISEME: dict[str, str] = {
    # Bilabial
    "P": "PP", "B": "PP", "M": "PP",
    # Labiodental
    "F": "FF", "V": "FF",
    # Dental
    "TH": "TH", "DH": "TH",
    # Alveolar stop
    "T": "DD", "D": "DD",
    # Sibilant
    "S": "SS", "Z": "SS",
    # Postalveolar
    "SH": "CH", "ZH": "CH", "CH": "CH", "JH": "CH",
    # Velar
    "K": "kk", "G": "kk", "NG": "kk",
    # Nasal/liquid
    "N": "nn", "L": "nn",
    # Rhotic
    "R": "RR",
    "ER0": "RR", "ER1": "RR", "ER2": "RR",
    # Open vowels
    "AA0": "aa", "AA1": "aa", "AA2": "aa",
    "AH0": "aa", "AH1": "aa", "AH2": "aa",
    "AE0": "aa", "AE1": "aa", "AE2": "aa",
    # Diphthongs that start open
    "AW0": "aa", "AW1": "aa", "AW2": "aa",
    "AY0": "aa", "AY1": "aa", "AY2": "aa",
    # Front spread
    "IY0": "EE", "IY1": "EE", "IY2": "EE",
    "EY0": "EE", "EY1": "EE", "EY2": "EE",
    "Y": "EE",
    # Mid front
    "IH0": "ih", "IH1": "ih", "IH2": "ih",
    "EH0": "ih", "EH1": "ih", "EH2": "ih",
    # Back rounded
    "AO0": "OH", "AO1": "OH", "AO2": "OH",
    "OW0": "OH", "OW1": "OH", "OW2": "OH",
    "OY0": "OH", "OY1": "OH", "OY2": "OH",
    # Close rounded
    "UW0": "OO", "UW1": "OO", "UW2": "OO",
    "UH0": "OO", "UH1": "OO", "UH2": "OO",
    "W": "OO",
    # Glottal / aspirate (map to open -- mouth opens for airflow)
    "HH": "aa",
}

# ---------------------------------------------------------------------------
# Natural-language prompt descriptions per viseme
# ---------------------------------------------------------------------------

VISEME_PROMPTS: dict[str, str] = {
    "sil": "lips gently closed, not speaking",
    "PP":  "lips pressed firmly together",
    "FF":  "lower lip tucked under upper teeth",
    "TH":  "tongue slightly visible between teeth",
    "DD":  "tongue touching behind upper teeth, mouth slightly open",
    "kk":  "mouth slightly open, back of tongue raised",
    "CH":  "lips slightly rounded and pushed forward",
    "SS":  "teeth close together, slight smile shape",
    "nn":  "mouth slightly open, tongue behind teeth",
    "RR":  "lips slightly rounded, tongue curled",
    "aa":  "mouth wide open, jaw dropped",
    "EE":  "lips spread wide, teeth visible",
    "ih":  "mouth slightly open, relaxed",
    "OH":  "lips rounded into oval shape",
    "OO":  "lips pursed into small round opening",
}

# ---------------------------------------------------------------------------
# FLAME mouth-related coefficient indices
# ---------------------------------------------------------------------------

# From expression_skills.py FLAME_TO_AU mapping -- these are the mouth/jaw AUs.
# Indices 0-9 cover jaw_open, lip_spread, lip_funneler, lip_pucker,
# lip_tightener, brow_raise, eye_squint, mouth_stretch, lip_press,
# lip_corner_pull.  We keep the first 10 expression + 3 jaw_pose axes.
MOUTH_EXPR_INDICES = list(range(10))
# Total mouth param dimensionality: 10 expression + 3 jaw_pose = 13
MOUTH_PARAM_DIM = 13


def _strip_stress(phoneme: str) -> str:
    """Strip trailing stress digit from a CMU phoneme (e.g. 'AA1' -> 'AA1' kept as-is for lookup)."""
    return phoneme


def _phoneme_to_viseme(phoneme: str) -> str:
    """Map a CMU phoneme to its viseme, with fallback."""
    # Direct lookup first
    vis = PHONEME_TO_VISEME.get(phoneme)
    if vis is not None:
        return vis
    # Try stripping stress digit
    base = re.sub(r"\d+$", "", phoneme)
    vis = PHONEME_TO_VISEME.get(base)
    if vis is not None:
        return vis
    # Unknown phoneme -> silence
    return "sil"


def word_to_phonemes(word: str, cmu: Optional[dict] = None) -> list[str]:
    """Look up phonemes for a word using CMU dict.

    Falls back to letter-by-letter mapping for unknown words.
    """
    if cmu is None:
        cmu = cmudict.dict()
    clean = word.lower().strip(".,!?;:\"'()-")
    if not clean:
        return []
    entries = cmu.get(clean)
    if entries:
        return entries[0]  # Use first pronunciation
    # Fallback: map each letter to a rough phoneme
    return _letters_to_phonemes(clean)


# Simple letter -> phoneme fallback for words not in CMU dict
_LETTER_PHONEMES: dict[str, str] = {
    "a": "AH0", "b": "B", "c": "K", "d": "D", "e": "EH0",
    "f": "F", "g": "G", "h": "HH", "i": "IH0", "j": "JH",
    "k": "K", "l": "L", "m": "M", "n": "N", "o": "OW0",
    "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
    "u": "AH0", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z",
}


def _letters_to_phonemes(word: str) -> list[str]:
    """Rough letter-by-letter phoneme approximation for unknown words."""
    return [_LETTER_PHONEMES.get(c, "AH0") for c in word.lower() if c.isalpha()]


def phonemes_to_visemes(phonemes: list[str]) -> list[str]:
    """Map a list of CMU phonemes to viseme names."""
    return [_phoneme_to_viseme(p) for p in phonemes]


# ---------------------------------------------------------------------------
# VisemeExtractor -- builds the viseme -> FLAME lookup table
# ---------------------------------------------------------------------------

class VisemeExtractor:
    """Extract viseme-to-FLAME-mouth-param lookup table from training data.

    Connects real word timestamps from the ClipCannon DB with FLAME
    expression vectors captured at each frame of the training video.
    """

    def __init__(
        self,
        db_path: str | pathlib.Path,
        flame_params_path: str | pathlib.Path,
    ) -> None:
        self.db_path = pathlib.Path(db_path)
        self.flame_path = pathlib.Path(flame_params_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")
        if not self.flame_path.exists():
            raise FileNotFoundError(f"FLAME params not found: {self.flame_path}")

        # Load FLAME data
        flame = np.load(str(self.flame_path))
        self._expression = flame["expression"]       # (N, 100)
        self._jaw_pose = flame["jaw_pose"]           # (N, 3)
        self._timestamps = flame["timestamps"]        # (N,) in seconds

        # Load words from DB
        self._words = self._load_words()

        # CMU dict
        self._cmu = cmudict.dict()

        # Viseme table (populated by extract_viseme_table)
        self._viseme_table: dict[str, np.ndarray] | None = None
        self._viseme_counts: dict[str, int] = {}

    def _load_words(self) -> list[dict]:
        """Load transcript words from ClipCannon DB."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT word, start_ms, end_ms FROM transcript_words ORDER BY start_ms"
        )
        words = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return words

    def _get_mouth_params_at_time(self, time_s: float) -> np.ndarray:
        """Get mouth-related FLAME params at a given timestamp.

        Finds the nearest FLAME frame and extracts the 13-dim mouth vector
        (10 expression coefficients + 3 jaw_pose axes).
        """
        idx = np.argmin(np.abs(self._timestamps - time_s))
        expr_mouth = self._expression[idx, MOUTH_EXPR_INDICES]
        jaw = self._jaw_pose[idx]
        return np.concatenate([expr_mouth, jaw])

    def _get_mouth_params_range(
        self, start_s: float, end_s: float
    ) -> np.ndarray:
        """Get average mouth params over a time range."""
        mask = (self._timestamps >= start_s) & (self._timestamps <= end_s)
        if not np.any(mask):
            # Fall back to nearest frame
            return self._get_mouth_params_at_time((start_s + end_s) / 2)
        expr_mouth = self._expression[mask][:, MOUTH_EXPR_INDICES]
        jaw = self._jaw_pose[mask]
        combined = np.concatenate([expr_mouth, jaw], axis=1)
        return combined.mean(axis=0)

    def extract_viseme_table(self) -> dict[str, np.ndarray]:
        """Build viseme -> average FLAME mouth params lookup table.

        For each word in the transcript:
          1. Look up phonemes in CMU dict
          2. Map phonemes to visemes
          3. Find FLAME expression at the word's timestamp range
          4. Extract mouth-related params (10 expression + 3 jaw)
          5. Accumulate and average all FLAME params for each viseme

        Returns:
            Dict mapping viseme name to averaged (13,) FLAME mouth param vector.
        """
        # Accumulators: viseme_name -> list of param vectors
        accum: dict[str, list[np.ndarray]] = {v: [] for v in VISEME_MAP}

        for word_rec in self._words:
            word = word_rec["word"]
            start_s = word_rec["start_ms"] / 1000.0
            end_s = word_rec["end_ms"] / 1000.0

            phonemes = word_to_phonemes(word, self._cmu)
            if not phonemes:
                continue

            visemes = phonemes_to_visemes(phonemes)
            n_phonemes = len(phonemes)

            # Distribute word duration evenly across its phonemes
            phone_dur = (end_s - start_s) / n_phonemes if n_phonemes > 0 else 0

            for i, vis in enumerate(visemes):
                phone_start = start_s + i * phone_dur
                phone_end = phone_start + phone_dur
                # Get the midpoint mouth params for this phoneme window
                phone_mid = (phone_start + phone_end) / 2.0
                params = self._get_mouth_params_at_time(phone_mid)
                accum[vis].append(params)

        # Also add silence samples from gaps between words
        for i in range(len(self._words) - 1):
            gap_start = self._words[i]["end_ms"] / 1000.0
            gap_end = self._words[i + 1]["start_ms"] / 1000.0
            gap_dur = gap_end - gap_start
            if gap_dur > 0.1:  # Only count gaps > 100ms as silence
                params = self._get_mouth_params_at_time(
                    (gap_start + gap_end) / 2.0
                )
                accum["sil"].append(params)

        # Average each viseme
        table: dict[str, np.ndarray] = {}
        for vis_name, param_list in accum.items():
            self._viseme_counts[vis_name] = len(param_list)
            if param_list:
                table[vis_name] = np.mean(param_list, axis=0).astype(np.float32)
            else:
                # No samples -- use neutral (zeros)
                table[vis_name] = np.zeros(MOUTH_PARAM_DIM, dtype=np.float32)

        self._viseme_table = table
        return table

    @property
    def viseme_table(self) -> dict[str, np.ndarray]:
        """Return the viseme table, extracting if not yet done."""
        if self._viseme_table is None:
            self.extract_viseme_table()
        return self._viseme_table

    @property
    def viseme_counts(self) -> dict[str, int]:
        """Number of samples accumulated for each viseme."""
        if not self._viseme_counts:
            self.extract_viseme_table()
        return self._viseme_counts

    def text_to_viseme_sequence(
        self,
        text: str,
        duration_s: float,
        fps: int = 25,
    ) -> list[tuple[int, str, np.ndarray]]:
        """Convert text to per-frame viseme sequence with FLAME params.

        Distributes words evenly across the duration, then distributes
        phonemes within each word's time window.

        Args:
            text: Input text to synthesize visemes for.
            duration_s: Total duration in seconds.
            fps: Target frame rate.

        Returns:
            List of (frame_idx, viseme_name, flame_mouth_params) tuples,
            one per frame.
        """
        table = self.viseme_table
        total_frames = int(duration_s * fps)
        if total_frames <= 0:
            return []

        words = text.split()
        if not words:
            silence_params = table.get("sil", np.zeros(MOUTH_PARAM_DIM, dtype=np.float32))
            return [(f, "sil", silence_params) for f in range(total_frames)]

        # Build phoneme timeline: list of (viseme_name, start_time, end_time)
        timeline: list[tuple[str, float, float]] = []
        word_dur = duration_s / len(words)

        for w_idx, word in enumerate(words):
            w_start = w_idx * word_dur
            w_end = w_start + word_dur

            phonemes = word_to_phonemes(word, self._cmu)
            if not phonemes:
                # Treat entire word as silence
                timeline.append(("sil", w_start, w_end))
                continue

            visemes = phonemes_to_visemes(phonemes)
            n_phones = len(visemes)
            phone_dur = (w_end - w_start) / n_phones

            for p_idx, vis in enumerate(visemes):
                p_start = w_start + p_idx * phone_dur
                p_end = p_start + phone_dur
                timeline.append((vis, p_start, p_end))

        # Generate per-frame output
        result: list[tuple[int, str, np.ndarray]] = []
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            # Find which timeline segment this frame falls into
            vis_name = "sil"
            for seg_vis, seg_start, seg_end in timeline:
                if seg_start <= t < seg_end:
                    vis_name = seg_vis
                    break

            params = table.get(vis_name, np.zeros(MOUTH_PARAM_DIM, dtype=np.float32))
            result.append((frame_idx, vis_name, params))

        return result

    def viseme_to_prompt(self, viseme: str) -> str:
        """Convert a viseme to a natural language mouth description."""
        return VISEME_PROMPTS.get(viseme, "mouth in neutral position")

    def save_table(self, output_path: str | pathlib.Path) -> None:
        """Save the viseme lookup table to an NPZ file.

        Saves arrays named by viseme (e.g. 'sil', 'PP', 'FF', ...) plus
        a 'viseme_names' metadata array and 'viseme_counts'.
        """
        table = self.viseme_table
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        save_dict: dict[str, np.ndarray] = {}
        names = []
        counts = []
        for vis_name in VISEME_MAP:
            save_dict[f"vis_{vis_name}"] = table[vis_name]
            names.append(vis_name)
            counts.append(self._viseme_counts.get(vis_name, 0))

        save_dict["viseme_names"] = np.array(names)
        save_dict["viseme_counts"] = np.array(counts, dtype=np.int32)

        np.savez_compressed(str(out), **save_dict)

    @classmethod
    def load_table(cls, npz_path: str | pathlib.Path) -> dict[str, np.ndarray]:
        """Load a previously saved viseme table from NPZ."""
        data = np.load(str(npz_path), allow_pickle=True)
        table: dict[str, np.ndarray] = {}
        names = data["viseme_names"]
        for name in names:
            key = str(name)
            table[key] = data[f"vis_{key}"]
        return table

    def get_statistics(self) -> dict:
        """Return extraction statistics."""
        _ = self.viseme_table  # Ensure extracted
        total_samples = sum(self._viseme_counts.values())
        coverage = sum(1 for c in self._viseme_counts.values() if c > 0)
        return {
            "total_words": len(self._words),
            "total_samples": total_samples,
            "visemes_covered": coverage,
            "visemes_total": len(VISEME_MAP),
            "coverage_pct": coverage / len(VISEME_MAP) * 100,
            "per_viseme": dict(self._viseme_counts),
        }
