"""Semantic Position Decoders (SPDs) -- translate raw embeddings to meaning.

Each SPD takes a raw embedding vector from one modality and outputs a
32-dimensional semantic position vector with human-interpretable channels.
Calibrated per-person on their actual data ranges.

Architecture (7 modalities):
  VisualSPD:   SigLIP 1152-dim   -> 32 semantic dims  (3-layer, hidden=128)
  EmotionSPD:  Wav2Vec2 1024-dim  -> 32 semantic dims  (3-layer, hidden=128)
  ProsodySPD:  prosody 12-dim     -> 32 semantic dims  (2-layer, hidden=64)
  SemanticSPD: Nomic 768-dim      -> 32 semantic dims  (3-layer, hidden=128)
  SpeakerSPD:  WavLM 512-dim      -> 32 semantic dims  (3-layer, hidden=128)
  SentenceSPD: MiniLM 384-dim     -> 32 semantic dims  (2-layer, hidden=64)
  VoiceSPD:    ECAPA 192-dim      -> 32 semantic dims  (2-layer, hidden=64)

Training: Self-supervised from ClipCannon analysis labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

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

SPEAKER_CHANNELS = [
    "voice_depth", "voice_brightness", "voice_roughness", "voice_breathiness",
    "nasality", "resonance", "vocal_fry", "falsetto",
    "age_young", "age_middle", "age_old", "gender_masculine",
    "gender_feminine", "accent_strength", "dialect_marker", "speech_clarity",
    "loudness_preference", "pitch_range", "articulation", "pace_natural",
    "pace_excited", "pace_calm", "register_formal", "register_casual",
    "authority", "warmth_vocal", "confidence_vocal", "nervousness",
    "tiredness", "health", "consistency", "distinctiveness",
]

SENTENCE_CHANNELS = [
    "declarative", "interrogative", "imperative", "exclamatory",
    "conditional", "hypothetical", "comparative", "superlative",
    "simple_structure", "compound", "complex_structure", "fragment",
    "short_utterance", "long_utterance", "filler_content", "substantive",
    "topic_intro", "topic_continuation", "topic_shift", "elaboration_sent",
    "example_giving", "summarizing", "quoting", "paraphrasing",
    "agreement_sent", "disagreement_sent", "hedging_sent", "emphasis_sent",
    "humor_sent", "irony", "metaphor", "literal",
]

VOICE_CHANNELS = [
    "fundamental_freq", "harmonic_richness", "spectral_tilt", "formant_spacing",
    "jitter", "shimmer", "hnr", "cepstral_peak",
    "chest_voice", "head_voice", "mixed_voice", "whisper_quality",
    "projected", "intimate", "resonant", "thin",
    "vowel_space", "consonant_precision", "coarticulation", "speaking_effort",
    "vocal_strain", "relaxed_production", "onset_sharp", "onset_soft",
    "timbre_warm", "timbre_bright", "timbre_dark", "timbre_metallic",
    "vibrato", "tremor", "glottal_quality", "airflow_quality",
]


@dataclass
class SPDConfig:
    """Configuration for Semantic Position Decoders."""
    hidden_dim: int = 128
    semantic_dim: int = SEMANTIC_DIM
    dropout: float = 0.1
    use_residual: bool = True


# ---------------------------------------------------------------------------
# Helper: build MLP for a given input size
# ---------------------------------------------------------------------------
def _build_large_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float,
) -> nn.Sequential:
    """3-layer MLP for larger inputs (>=512d)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _build_small_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float,
) -> nn.Sequential:
    """2-layer MLP for smaller inputs (<512d)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


# ---------------------------------------------------------------------------
# Individual SPDs
# ---------------------------------------------------------------------------
class VisualSPD(nn.Module):
    """Decode SigLIP 1152-dim visual embeddings to 32 semantic dims."""

    INPUT_DIM = 1152

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = _build_large_mlp(self.INPUT_DIM, 128, cfg.semantic_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        out = self.net(x_norm)
        return torch.sigmoid(self.norm(out))

    def calibrate(self, data: np.ndarray) -> None:
        """Set normalization from training data statistics."""
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class EmotionSPD(nn.Module):
    """Decode Wav2Vec2 1024-dim hidden state to 32 semantic dims."""

    INPUT_DIM = 1024

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = _build_large_mlp(self.INPUT_DIM, 128, cfg.semantic_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class ProsodySPD(nn.Module):
    """Decode 12-dim prosody features to 32 semantic dims."""

    INPUT_DIM = 12

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = _build_small_mlp(self.INPUT_DIM, 64, cfg.semantic_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class SemanticSPD(nn.Module):
    """Decode Nomic 768-dim text embeddings to 32 semantic dims."""

    INPUT_DIM = 768

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = _build_large_mlp(self.INPUT_DIM, 128, cfg.semantic_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class SpeakerSPD(nn.Module):
    """Decode WavLM 512-dim speaker embeddings to 32 semantic dims."""

    INPUT_DIM = 512

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = _build_large_mlp(self.INPUT_DIM, 128, cfg.semantic_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class SentenceSPD(nn.Module):
    """Decode MiniLM 384-dim sentence embeddings to 32 semantic dims."""

    INPUT_DIM = 384

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = _build_small_mlp(self.INPUT_DIM, 64, cfg.semantic_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-6)
        return torch.sigmoid(self.norm(self.net(x_norm)))

    def calibrate(self, data: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(data.mean(axis=0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(data.std(axis=0).astype(np.float32).clip(1e-6)))


class VoiceSPD(nn.Module):
    """Decode ECAPA 192-dim voice embeddings to 32 semantic dims."""

    INPUT_DIM = 192

    def __init__(self, config: SPDConfig | None = None) -> None:
        super().__init__()
        cfg = config or SPDConfig()
        self.net = _build_small_mlp(self.INPUT_DIM, 64, cfg.semantic_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.semantic_dim)
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))

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

    return np.clip(labels, 0, 1).astype(np.float32)


def generate_emotion_labels(emo_emb: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for emotion SPD from Wav2Vec2 1024-dim hidden state.

    Uses quadrant analysis and statistical proxies to derive emotion labels.

    Args:
        emo_emb: (N, 1024) float32 array -- Wav2Vec2 mean-pooled hidden states.

    Returns:
        (N, 32) float32 labels in [0, 1].

    Raises:
        ValueError: If input does not have 1024 dimensions.
    """
    if emo_emb.ndim != 2 or emo_emb.shape[1] != 1024:
        raise ValueError(
            f"generate_emotion_labels expects (N, 1024) input, got {emo_emb.shape}"
        )

    N = len(emo_emb)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        emb = emo_emb[i]
        norm = float(np.linalg.norm(emb))
        var = float(np.var(emb))
        mean_val = float(np.mean(emb))

        # Quadrant analysis of 1024-dim vector
        q1 = float(emb[:256].mean())
        q2 = float(emb[256:512].mean())
        q3 = float(emb[512:768].mean())
        q4 = float(emb[768:].mean())

        # Arousal proxy from variance (higher var = more aroused)
        arousal = min(1.0, max(0.0, var * 50.0))
        # Valence proxy from mean (shifted/scaled to [0,1])
        valence = min(1.0, max(0.0, mean_val * 5.0 + 0.5))
        # Energy proxy from L2 norm
        energy = min(1.0, max(0.0, norm / 10.0))
        # Dominance proxy from q4 quadrant
        dominance = min(1.0, max(0.0, q4 * 10.0 + 0.5))

        labels[i, 0] = max(0.0, valence * arousal)  # joy
        labels[i, 1] = max(0.0, (1.0 - valence) * 0.3)  # sadness
        labels[i, 2] = max(0.0, arousal * energy)  # excitement
        labels[i, 3] = max(0.0, (1.0 - arousal) * (1.0 - energy))  # calm
        labels[i, 4] = max(0.0, valence * 0.7)  # warmth
        labels[i, 5] = max(0.0, (1.0 - valence) * arousal * 0.5)  # concern
        labels[i, 6] = max(0.0, q1 * 5.0 + 0.3)  # amusement
        labels[i, 7] = max(0.0, (1.0 - arousal) * 0.5)  # thoughtfulness
        labels[i, 8] = max(0.0, min(1.0, abs(q2) * 10.0))  # surprise
        labels[i, 9] = max(0.0, valence * (1.0 - dominance) * 0.6)  # empathy
        labels[i, 10] = max(0.0, dominance * valence * 0.5)  # pride
        labels[i, 11] = max(0.0, valence * 0.4)  # gratitude
        labels[i, 12] = max(0.0, q3 * 5.0 + 0.3)  # nostalgia
        labels[i, 13] = max(0.0, arousal * 0.6)  # anticipation
        labels[i, 14] = max(0.0, valence * (1.0 - arousal) * 0.8)  # contentment
        labels[i, 15] = max(0.0, (1.0 - valence) * arousal * 0.4)  # frustration
        labels[i, 16] = arousal  # arousal
        labels[i, 17] = valence  # valence
        labels[i, 18] = energy  # energy
        labels[i, 19] = dominance  # dominance
        labels[i, 20] = max(0.0, valence * 0.8)  # positive_affect
        labels[i, 21] = max(0.0, (1.0 - valence) * 0.5)  # negative_affect
        labels[i, 22] = max(0.0, min(1.0, abs(q1 - q3) * 5.0))  # mixed_emotion
        labels[i, 23] = (arousal + energy) / 2.0  # intensity
        labels[i, 24] = max(0.0, 1.0 - abs(q2 - q4) * 5.0)  # authenticity
        labels[i, 25] = max(0.0, 1.0 - var * 30.0)  # congruence
        labels[i, 26] = max(0.0, min(1.0, var * 40.0))  # transition_speed
        labels[i, 27] = max(0.0, 1.0 - var * 20.0)  # stability
        labels[i, 28] = max(0.0, valence * 0.8)  # social_warmth
        labels[i, 29] = max(0.0, (1.0 - dominance) * arousal * 0.4)  # vulnerability
        labels[i, 30] = max(0.0, dominance * 0.7)  # confidence
        labels[i, 31] = max(0.0, valence * arousal * 0.6)  # playfulness

    return np.clip(labels, 0, 1).astype(np.float32)


def generate_prosody_labels(pro_data: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for prosody SPD from prosody features.

    Args:
        pro_data: (N, 12) float32 array -- prosody feature vectors.

    Returns:
        (N, 32) float32 labels in [0, 1].

    Raises:
        ValueError: If input does not have 12 dimensions.
    """
    if pro_data.ndim != 2 or pro_data.shape[1] != 12:
        raise ValueError(
            f"generate_prosody_labels expects (N, 12) input, got {pro_data.shape}"
        )

    N = len(pro_data)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        f = pro_data[i]
        # Raw prosody values need normalization to [0,1]:
        # col 0: f0_mean ~100-250 Hz, col 1: f0_range ~5-110 Hz
        # col 2: energy ~60-150, col 3: speaking_rate ~100-600 WPM
        # col 4: emphasis ~25-510, col 5-7: small floats ~0-0.2
        f0_norm = f[0] / 300.0
        f0_range = f[1] / 150.0
        energy = f[2] / 200.0
        rate = f[3] / 600.0
        emphasis = f[4] / 600.0
        rising = f[5] / 0.1 if f[5] < 0.1 else 1.0

        labels[i, 0] = 1.0 if rate > 0.05 else 0.0  # speaking
        labels[i, 1] = min(1.0, emphasis)  # emphatic
        labels[i, 2] = min(1.0, rising)  # questioning
        labels[i, 4] = max(0.0, 1.0 - energy)  # calm_speaking
        labels[i, 5] = min(1.0, energy)  # energetic_speaking
        labels[i, 6] = max(0.0, 1.0 - rate * 2)  # slow_rate
        labels[i, 7] = min(1.0, rate * 1.5)  # fast_rate
        labels[i, 8] = min(1.0, rising)  # rising_pitch
        labels[i, 9] = max(0.0, 1.0 - rising)  # falling_pitch
        labels[i, 10] = min(1.0, f0_range * 2)  # varied_pitch
        labels[i, 20] = min(1.0, f0_norm)  # f0_relative
        labels[i, 21] = min(1.0, f0_range)  # f0_range_relative
        labels[i, 22] = min(1.0, energy)  # energy_relative
        labels[i, 23] = min(1.0, rate)  # rate_relative

    return np.clip(labels, 0, 1).astype(np.float32)


def generate_semantic_labels(sem_emb: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for semantic SPD from Nomic embeddings.

    Uses embedding structure as proxy for content type.

    Args:
        sem_emb: (N, 768) float32 array -- Nomic text embeddings.

    Returns:
        (N, 32) float32 labels in [0, 1].

    Raises:
        ValueError: If input does not have 768 dimensions.
    """
    if sem_emb.ndim != 2 or sem_emb.shape[1] != 768:
        raise ValueError(
            f"generate_semantic_labels expects (N, 768) input, got {sem_emb.shape}"
        )

    N = len(sem_emb)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        emb = sem_emb[i]
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


def generate_speaker_labels(spk_emb: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for speaker SPD from WavLM 512-dim embeddings.

    Uses quadrant analysis, L2 norm, and variance to derive speaker identity labels.

    Args:
        spk_emb: (N, 512) float32 array -- WavLM speaker embeddings.

    Returns:
        (N, 32) float32 labels in [0, 1].

    Raises:
        ValueError: If input does not have 512 dimensions.
    """
    if spk_emb.ndim != 2 or spk_emb.shape[1] != 512:
        raise ValueError(
            f"generate_speaker_labels expects (N, 512) input, got {spk_emb.shape}"
        )

    N = len(spk_emb)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        emb = spk_emb[i]
        norm = float(np.linalg.norm(emb))
        var = float(np.var(emb))

        # Quadrant analysis
        q1 = float(emb[:128].mean())
        q2 = float(emb[128:256].mean())
        q3 = float(emb[256:384].mean())
        q4 = float(emb[384:].mean())

        # L2 norm as proxy for speaker distinctiveness
        distinctiveness = min(1.0, max(0.0, norm / 8.0))
        # Variance across dims as proxy for voice complexity
        complexity = min(1.0, max(0.0, var * 30.0))

        # Voice quality features from quadrants
        labels[i, 0] = max(0.0, min(1.0, q1 * 5.0 + 0.5))  # voice_depth
        labels[i, 1] = max(0.0, min(1.0, -q1 * 5.0 + 0.5))  # voice_brightness
        labels[i, 2] = max(0.0, min(1.0, abs(q2) * 8.0))  # voice_roughness
        labels[i, 3] = max(0.0, min(1.0, (1.0 - norm / 10.0)))  # voice_breathiness
        labels[i, 4] = max(0.0, min(1.0, q3 * 5.0 + 0.5))  # nasality
        labels[i, 5] = max(0.0, min(1.0, norm / 12.0))  # resonance
        labels[i, 6] = max(0.0, min(1.0, abs(q4) * 6.0))  # vocal_fry
        labels[i, 7] = max(0.0, min(1.0, max(0, -q4) * 8.0))  # falsetto

        # Demographics from embedding structure
        labels[i, 8] = max(0.0, min(1.0, q2 * 4.0 + 0.3))  # age_young
        labels[i, 9] = max(0.0, min(1.0, 0.5 - abs(q2) * 3.0))  # age_middle
        labels[i, 10] = max(0.0, min(1.0, -q2 * 4.0 + 0.3))  # age_old
        labels[i, 11] = max(0.0, min(1.0, q1 * 4.0 + 0.5))  # gender_masculine
        labels[i, 12] = max(0.0, min(1.0, -q1 * 4.0 + 0.5))  # gender_feminine
        labels[i, 13] = max(0.0, min(1.0, abs(q3 - q1) * 5.0))  # accent_strength
        labels[i, 14] = max(0.0, min(1.0, abs(q4 - q2) * 5.0))  # dialect_marker
        labels[i, 15] = max(0.0, min(1.0, norm / 10.0))  # speech_clarity

        # Speaking style
        labels[i, 16] = max(0.0, min(1.0, norm / 12.0))  # loudness_preference
        labels[i, 17] = complexity  # pitch_range
        labels[i, 18] = max(0.0, min(1.0, norm / 10.0))  # articulation
        labels[i, 19] = max(0.0, min(1.0, 0.5 + q3 * 3.0))  # pace_natural
        labels[i, 20] = max(0.0, min(1.0, complexity * 0.8))  # pace_excited
        labels[i, 21] = max(0.0, min(1.0, (1.0 - complexity) * 0.7))  # pace_calm
        labels[i, 22] = max(0.0, min(1.0, -q4 * 5.0 + 0.5))  # register_formal
        labels[i, 23] = max(0.0, min(1.0, q4 * 5.0 + 0.5))  # register_casual

        # Personality proxies
        labels[i, 24] = max(0.0, min(1.0, norm / 10.0 * 0.8))  # authority
        labels[i, 25] = max(0.0, min(1.0, q3 * 5.0 + 0.5))  # warmth_vocal
        labels[i, 26] = max(0.0, min(1.0, distinctiveness * 0.9))  # confidence_vocal
        labels[i, 27] = max(0.0, min(1.0, var * 40.0))  # nervousness
        labels[i, 28] = max(0.0, min(1.0, (1.0 - norm / 8.0)))  # tiredness
        labels[i, 29] = max(0.0, min(1.0, norm / 8.0))  # health
        labels[i, 30] = max(0.0, min(1.0, 1.0 - var * 20.0))  # consistency
        labels[i, 31] = distinctiveness  # distinctiveness

    return np.clip(labels, 0, 1).astype(np.float32)


def generate_sentence_labels(sent_emb: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for sentence SPD from MiniLM 384-dim embeddings.

    Uses embedding norm, quadrant means, and variance to derive sentence type labels.

    Args:
        sent_emb: (N, 384) float32 array -- MiniLM sentence embeddings.

    Returns:
        (N, 32) float32 labels in [0, 1].

    Raises:
        ValueError: If input does not have 384 dimensions.
    """
    if sent_emb.ndim != 2 or sent_emb.shape[1] != 384:
        raise ValueError(
            f"generate_sentence_labels expects (N, 384) input, got {sent_emb.shape}"
        )

    N = len(sent_emb)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        emb = sent_emb[i]
        norm = float(np.linalg.norm(emb))
        var = float(np.var(emb))
        mean_val = float(np.mean(emb))

        # Quadrant analysis (4 x 96-dim chunks)
        q1 = float(emb[:96].mean())
        q2 = float(emb[96:192].mean())
        q3 = float(emb[192:288].mean())
        q4 = float(emb[288:].mean())

        # Norm as proxy for content richness
        richness = min(1.0, max(0.0, norm / 6.0))
        # Variance as proxy for complexity
        complexity = min(1.0, max(0.0, var * 40.0))

        # Sentence type features
        labels[i, 0] = max(0.0, min(1.0, q1 * 5.0 + 0.5))  # declarative
        labels[i, 1] = max(0.0, min(1.0, -q1 * 5.0 + 0.3))  # interrogative
        labels[i, 2] = max(0.0, min(1.0, q2 * 5.0 + 0.2))  # imperative
        labels[i, 3] = max(0.0, min(1.0, abs(q2) * 8.0))  # exclamatory
        labels[i, 4] = max(0.0, min(1.0, q3 * 4.0 + 0.3))  # conditional
        labels[i, 5] = max(0.0, min(1.0, -q3 * 4.0 + 0.2))  # hypothetical
        labels[i, 6] = max(0.0, min(1.0, abs(q4 - q1) * 5.0))  # comparative
        labels[i, 7] = max(0.0, min(1.0, abs(q4) * 6.0))  # superlative

        # Structure features
        labels[i, 8] = max(0.0, min(1.0, 1.0 - complexity))  # simple_structure
        labels[i, 9] = max(0.0, min(1.0, complexity * 0.7))  # compound
        labels[i, 10] = complexity  # complex_structure
        labels[i, 11] = max(0.0, min(1.0, 1.0 - richness))  # fragment
        labels[i, 12] = max(0.0, min(1.0, 1.0 - norm / 5.0))  # short_utterance
        labels[i, 13] = max(0.0, min(1.0, norm / 8.0))  # long_utterance
        labels[i, 14] = max(0.0, min(1.0, 1.0 - richness * 0.8))  # filler_content
        labels[i, 15] = richness  # substantive

        # Discourse features
        labels[i, 16] = max(0.0, min(1.0, q4 * 4.0 + 0.3))  # topic_intro
        labels[i, 17] = max(0.0, min(1.0, 0.5 - abs(mean_val) * 5.0))  # topic_continuation
        labels[i, 18] = max(0.0, min(1.0, abs(q1 - q4) * 6.0))  # topic_shift
        labels[i, 19] = max(0.0, min(1.0, richness * 0.7))  # elaboration_sent
        labels[i, 20] = max(0.0, min(1.0, q3 * 4.0 + 0.3))  # example_giving
        labels[i, 21] = max(0.0, min(1.0, -q3 * 3.0 + 0.3))  # summarizing
        labels[i, 22] = max(0.0, min(1.0, abs(q2 - q3) * 5.0))  # quoting
        labels[i, 23] = max(0.0, min(1.0, abs(q1 - q3) * 4.0))  # paraphrasing

        # Pragmatic features
        labels[i, 24] = max(0.0, min(1.0, q1 * 3.0 + 0.4))  # agreement_sent
        labels[i, 25] = max(0.0, min(1.0, -q1 * 3.0 + 0.2))  # disagreement_sent
        labels[i, 26] = max(0.0, min(1.0, (1.0 - richness) * 0.6))  # hedging_sent
        labels[i, 27] = max(0.0, min(1.0, abs(q4) * 6.0))  # emphasis_sent
        labels[i, 28] = max(0.0, min(1.0, q2 * 4.0 + 0.2))  # humor_sent
        labels[i, 29] = max(0.0, min(1.0, abs(q2 - q4) * 5.0))  # irony
        labels[i, 30] = max(0.0, min(1.0, abs(q3 - q1) * 4.0))  # metaphor
        labels[i, 31] = max(0.0, min(1.0, 1.0 - abs(q3 - q1) * 4.0))  # literal

    return np.clip(labels, 0, 1).astype(np.float32)


def generate_voice_labels(voice_emb: np.ndarray) -> np.ndarray:
    """Generate 32-dim pseudo-labels for voice SPD from ECAPA 192-dim embeddings.

    Uses quadrant analysis and norm heuristics focused on timbre features.
    ECAPA captures speaker identity so labels focus on voice quality/timbre.

    Args:
        voice_emb: (N, 192) float32 array -- ECAPA voice embeddings.

    Returns:
        (N, 32) float32 labels in [0, 1].

    Raises:
        ValueError: If input does not have 192 dimensions.
    """
    if voice_emb.ndim != 2 or voice_emb.shape[1] != 192:
        raise ValueError(
            f"generate_voice_labels expects (N, 192) input, got {voice_emb.shape}"
        )

    N = len(voice_emb)
    labels = np.zeros((N, SEMANTIC_DIM), dtype=np.float32)

    for i in range(N):
        emb = voice_emb[i]
        norm = float(np.linalg.norm(emb))
        var = float(np.var(emb))

        # Quadrant analysis (4 x 48-dim chunks)
        q1 = float(emb[:48].mean())
        q2 = float(emb[48:96].mean())
        q3 = float(emb[96:144].mean())
        q4 = float(emb[144:].mean())

        # Acoustic features
        labels[i, 0] = max(0.0, min(1.0, q1 * 5.0 + 0.5))  # fundamental_freq
        labels[i, 1] = max(0.0, min(1.0, norm / 6.0))  # harmonic_richness
        labels[i, 2] = max(0.0, min(1.0, q2 * 5.0 + 0.5))  # spectral_tilt
        labels[i, 3] = max(0.0, min(1.0, abs(q1 - q3) * 6.0))  # formant_spacing
        labels[i, 4] = max(0.0, min(1.0, var * 30.0))  # jitter
        labels[i, 5] = max(0.0, min(1.0, var * 25.0))  # shimmer
        labels[i, 6] = max(0.0, min(1.0, norm / 8.0))  # hnr
        labels[i, 7] = max(0.0, min(1.0, abs(q4) * 6.0))  # cepstral_peak

        # Register features
        labels[i, 8] = max(0.0, min(1.0, q1 * 4.0 + 0.5))  # chest_voice
        labels[i, 9] = max(0.0, min(1.0, -q1 * 4.0 + 0.3))  # head_voice
        labels[i, 10] = max(0.0, min(1.0, 0.5 - abs(q1) * 3.0))  # mixed_voice
        labels[i, 11] = max(0.0, min(1.0, (1.0 - norm / 5.0)))  # whisper_quality
        labels[i, 12] = max(0.0, min(1.0, norm / 6.0))  # projected
        labels[i, 13] = max(0.0, min(1.0, (1.0 - norm / 6.0)))  # intimate
        labels[i, 14] = max(0.0, min(1.0, norm / 7.0))  # resonant
        labels[i, 15] = max(0.0, min(1.0, (1.0 - norm / 5.0)))  # thin

        # Articulation features
        labels[i, 16] = max(0.0, min(1.0, abs(q3) * 6.0))  # vowel_space
        labels[i, 17] = max(0.0, min(1.0, norm / 7.0))  # consonant_precision
        labels[i, 18] = max(0.0, min(1.0, abs(q2 - q3) * 5.0))  # coarticulation
        labels[i, 19] = max(0.0, min(1.0, norm / 5.0))  # speaking_effort
        labels[i, 20] = max(0.0, min(1.0, var * 40.0))  # vocal_strain
        labels[i, 21] = max(0.0, min(1.0, 1.0 - var * 30.0))  # relaxed_production
        labels[i, 22] = max(0.0, min(1.0, q4 * 5.0 + 0.4))  # onset_sharp
        labels[i, 23] = max(0.0, min(1.0, -q4 * 5.0 + 0.4))  # onset_soft

        # Timbre features
        labels[i, 24] = max(0.0, min(1.0, q3 * 4.0 + 0.5))  # timbre_warm
        labels[i, 25] = max(0.0, min(1.0, -q3 * 4.0 + 0.3))  # timbre_bright
        labels[i, 26] = max(0.0, min(1.0, q1 * 3.0 + 0.4))  # timbre_dark
        labels[i, 27] = max(0.0, min(1.0, abs(q2) * 5.0))  # timbre_metallic
        labels[i, 28] = max(0.0, min(1.0, var * 35.0))  # vibrato
        labels[i, 29] = max(0.0, min(1.0, var * 20.0))  # tremor
        labels[i, 30] = max(0.0, min(1.0, abs(q4 - q2) * 5.0))  # glottal_quality
        labels[i, 31] = max(0.0, min(1.0, (1.0 - abs(q4)) * 0.8))  # airflow_quality

    return np.clip(labels, 0, 1).astype(np.float32)
