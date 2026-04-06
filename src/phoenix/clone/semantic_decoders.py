"""Semantic Position Decoders (SPDs) -- translate raw embeddings to meaning.

Each SPD takes a raw embedding vector from one modality and outputs a
32-dimensional semantic position vector with human-interpretable channels.
Calibrated per-person on their actual data ranges.

Architecture:
  VisualSPD:   SigLIP 1152-dim  -> 32 semantic dims
  EmotionSPD:  emotion 3-dim    -> 32 semantic dims
  ProsodySPD:  prosody 12-dim   -> 32 semantic dims
  SemanticSPD: Nomic 768-dim    -> 32 semantic dims

Training: Self-supervised from ClipCannon analysis labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SEMANTIC_DIM = 32


# ---------------------------------------------------------------------------
# Channel definitions -- what each output dimension means
# ---------------------------------------------------------------------------
VISUAL_CHANNELS = [
    "face_visible", "expression_intensity", "mouth_open", "looking_at_camera",
    "head_angle", "smile_amount", "brow_raise", "squint",
    "eye_openness", "jaw_forward", "lip_pucker", "lip_spread",
    "head_tilt_lr", "head_tilt_ud", "lighting_front", "lighting_side",
    "occlusion", "blur", "distance_to_camera", "skin_tone_warm",
    "beard_visible", "glasses_visible", "hat_visible", "gesture_hand",
    "symmetry", "expression_velocity", "micro_expression", "dominant_emotion",
    "arousal_visual", "valence_visual", "engagement", "animation_level",
]

EMOTION_CHANNELS = [
    "joy", "sadness", "excitement", "calm",
    "warmth", "concern", "amusement", "thoughtfulness",
    "surprise", "empathy", "pride", "gratitude",
    "nostalgia", "anticipation", "contentment", "frustration",
    "arousal", "valence", "energy", "dominance",
    "positive_affect", "negative_affect", "mixed_emotion", "intensity",
    "authenticity", "congruence", "transition_speed", "stability",
    "social_warmth", "vulnerability", "confidence", "playfulness",
]

PROSODY_CHANNELS = [
    "speaking", "emphatic", "questioning", "storytelling",
    "calm_speaking", "energetic_speaking", "slow_rate", "fast_rate",
    "rising_pitch", "falling_pitch", "varied_pitch", "monotone",
    "loud", "soft", "breathy", "clear",
    "pausing", "hesitating", "laughing", "sighing",
    "f0_relative", "f0_range_relative", "energy_relative", "rate_relative",
    "rhythm_regular", "rhythm_irregular", "emphasis_strength", "phrase_final",
    "turn_taking", "backchanneling", "filler_words", "silence_comfort",
]

SEMANTIC_CHANNELS = [
    "topic_personal", "topic_factual", "topic_emotional", "topic_narrative",
    "sentiment_positive", "sentiment_negative", "sentiment_neutral", "formality",
    "specificity", "abstractness", "temporal_past", "temporal_present",
    "temporal_future", "certainty", "hedging", "humor",
    "reference_self", "reference_other", "reference_shared", "question_type",
    "answer_type", "greeting", "farewell", "acknowledgment",
    "elaboration", "summary", "redirect", "agreement",
    "disagreement", "suggestion", "instruction", "storytelling_semantic",
]


@dataclass
class SPDConfig:
    """Configuration for Semantic Position Decoders."""
    hidden_dim: int = 128
    semantic_dim: int = SEMANTIC_DIM
    dropout: float = 0.1
    use_residual: bool = True


# ---------------------------------------------------------------------------
# Individual SPDs
# ---------------------------------------------------------------------------
class VisualSPD(nn.Module):
    """Decode SigLIP 1152-dim visual embeddings to 32 semantic dims."""

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = nn.Sequential(
            nn.Linear(1152, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.semantic_dim),
        )
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        # Calibration buffers -- set from training data
        self.register_buffer("input_mean", torch.zeros(1152))
        self.register_buffer("input_std", torch.ones(1152))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        out = self.net(x_norm)
        return torch.sigmoid(self.norm(out))

    def calibrate(self, data: np.ndarray) -> None:
        """Set normalization from training data statistics."""
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class EmotionSPD(nn.Module):
    """Decode 3-dim emotion (arousal/valence/energy) to 32 semantic dims."""

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.semantic_dim),
        )
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(3))
        self.register_buffer("input_std", torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class ProsodySPD(nn.Module):
    """Decode 12-dim prosody features to 32 semantic dims."""

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.semantic_dim),
        )
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(12))
        self.register_buffer("input_std", torch.ones(12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class SemanticSPD(nn.Module):
    """Decode Nomic 768-dim text embeddings to 32 semantic dims."""

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = nn.Sequential(
            nn.Linear(768, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.semantic_dim),
        )
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(768))
        self.register_buffer("input_std", torch.ones(768))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


# ---------------------------------------------------------------------------
# Pseudo-label generation from ClipCannon analysis
# ---------------------------------------------------------------------------
def generate_visual_labels(
    vis_emb: np.ndarray,
    flame_exp: np.ndarray,
    flame_ts: np.ndarray,
    vis_ts: np.ndarray,
) -> np.ndarray:
    """Generate 32-dim pseudo-labels for visual SPD from FLAME + heuristics.

    Uses FLAME expression params as proxy for facial state.
    Returns (N, 32) float32 labels in [0, 1].
    """
    from phoenix.clone.meaning_trainer import _nearest_idx

    N = len(vis_emb)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        fidx = _nearest_idx(flame_ts, vis_ts[i])
        exp = flame_exp[fidx]
        exp_energy = np.abs(exp[:20]).mean()

        labels[i, 0] = 1.0  # face_visible (assumed in dataset)
        labels[i, 1] = min(1.0, exp_energy / 3.0)  # expression_intensity
        labels[i, 2] = min(1.0, max(0, exp[0]) / 5.0)  # mouth_open
        labels[i, 3] = 0.8  # looking_at_camera (interview)
        labels[i, 5] = min(1.0, max(0, (exp[1] + exp[6]) / 4.0))  # smile
        labels[i, 6] = min(1.0, max(0, exp[5]) / 3.0)  # brow_raise
        labels[i, 7] = min(1.0, max(0, exp[6]) / 2.0)  # squint
        labels[i, 8] = 0.7  # eye_openness default
        labels[i, 11] = min(1.0, max(0, exp[1]) / 3.0)  # lip_spread
        labels[i, 31] = min(1.0, exp_energy / 2.0)  # animation_level

    return labels


def generate_emotion_labels(emo_data: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for emotion SPD from arousal/valence/energy."""
    N = len(emo_data)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        a, v, e = emo_data[i]
        # Normalize to Santa's observed ranges
        a_n = np.clip((a - 0.15) / 0.07, 0, 1)
        v_n = np.clip((v - 0.50) / 0.01, 0, 1)
        e_n = np.clip((e - 0.29) / 0.06, 0, 1)

        labels[i, 0] = v_n * a_n  # joy
        labels[i, 1] = (1 - v_n) * 0.3  # sadness
        labels[i, 2] = a_n * e_n  # excitement
        labels[i, 3] = (1 - a_n) * (1 - e_n)  # calm
        labels[i, 4] = v_n * 0.7  # warmth
        labels[i, 7] = (1 - a_n) * 0.5  # thoughtfulness
        labels[i, 16] = a_n  # arousal
        labels[i, 17] = v_n  # valence
        labels[i, 18] = e_n  # energy
        labels[i, 23] = (a_n + e_n) / 2  # intensity
        labels[i, 28] = v_n * 0.8  # social_warmth

    return labels


def generate_prosody_labels(pro_data: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for prosody SPD from prosody features."""
    N = len(pro_data)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        f = pro_data[i]
        f0_norm = f[0]    # f0_mean/300
        f0_range = f[1]   # f0_range/500
        energy = f[2]     # energy_rms
        rate = f[3]       # speaking_rate/300
        emphasis = f[4]   # has_emphasis
        rising = f[5]     # pitch_rising

        labels[i, 0] = 1.0 if rate > 0.1 else 0.0  # speaking
        labels[i, 1] = emphasis  # emphatic
        labels[i, 2] = rising  # questioning
        labels[i, 4] = max(0, 1.0 - energy * 20)  # calm_speaking
        labels[i, 5] = min(1.0, energy * 20)  # energetic_speaking
        labels[i, 6] = max(0, 1.0 - rate * 2)  # slow_rate
        labels[i, 7] = min(1.0, rate * 1.5)  # fast_rate
        labels[i, 8] = rising  # rising_pitch
        labels[i, 9] = max(0, 1.0 - rising)  # falling_pitch
        labels[i, 10] = min(1.0, f0_range * 2)  # varied_pitch
        labels[i, 20] = f0_norm  # f0_relative
        labels[i, 21] = f0_range  # f0_range_relative
        labels[i, 22] = min(1.0, energy * 20)  # energy_relative
        labels[i, 23] = rate  # rate_relative

    return labels


def generate_semantic_labels(sem_emb: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for semantic SPD from Nomic embeddings.

    Uses embedding structure as proxy for content type.
    """
    N = len(sem_emb)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        emb = sem_emb[i]
        norm = np.linalg.norm(emb)
        # Quadrant-based heuristics
        q1 = emb[:192].mean()
        q2 = emb[192:384].mean()
        q3 = emb[384:576].mean()
        q4 = emb[576:].mean()

        labels[i, 0] = max(0, q1 * 10)  # topic_personal
        labels[i, 1] = max(0, -q1 * 10)  # topic_factual
        labels[i, 3] = max(0, q2 * 10)  # topic_narrative
        labels[i, 4] = max(0, q3 * 5 + 0.5)  # sentiment_positive
        labels[i, 5] = max(0, -q3 * 5)  # sentiment_negative
        labels[i, 6] = max(0, 1.0 - abs(q3) * 10)  # sentiment_neutral
        labels[i, 13] = max(0, q4 * 5 + 0.5)  # certainty

    return np.clip(labels, 0, 1)
