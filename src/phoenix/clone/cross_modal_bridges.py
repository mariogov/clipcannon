"""Cross-Modal Meaning Bridges (CMBs) -- encode physics between 7 modalities.

CMBs are small linear projections trained on paired data from a specific
person. They encode the PHYSICS of how modalities co-vary:
  - smile (visual) <-> high F0 (prosody)
  - open jaw (visual) <-> low F1 (prosody)
  - emphasis (prosody) <-> furrowed brows (visual)
  - same speaker = consistent voice (speaker <-> voice)

Trained once, frozen forever. These are physics, not learned correlations.

Architecture: 21 bidirectional bridges between 7 SPD spaces (C(7,2)=21 pairs).
Each bridge is a linear 32 -> 32 projection with physics-informed initialization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SEMANTIC_DIM = 32

MODALITY_NAMES = ["visual", "emotion", "prosody", "semantic", "speaker", "sentence", "voice"]

# Physics-informed initialization scales for each pair.
# Strong = 0.3: near identity, big initial signal
# Medium = 0.1: near identity, moderate signal
# Weak   = 0.01: near zero, let training decide
BRIDGE_PHYSICS: dict[tuple[str, str], float] = {
    ("visual", "emotion"): 0.3,     # Strong -- facial expression maps to emotional state
    ("visual", "prosody"): 0.1,     # Medium -- mouth movement correlates with speaking
    ("visual", "semantic"): 0.1,    # Medium -- topics correlate with expressions
    ("visual", "speaker"): 0.01,    # Weak -- appearance loosely correlates to identity
    ("visual", "sentence"): 0.01,   # Weak -- content doesn't strongly affect appearance
    ("visual", "voice"): 0.01,      # Weak -- face structure loosely correlates timbre
    ("emotion", "prosody"): 0.3,    # Strong -- emotional state drives prosodic features
    ("emotion", "semantic"): 0.1,   # Medium -- emotional content in speech
    ("emotion", "speaker"): 0.01,   # Weak -- identity is independent of momentary emotion
    ("emotion", "sentence"): 0.1,   # Medium -- sentence meaning carries emotional weight
    ("emotion", "voice"): 0.01,     # Weak -- voice timbre stable regardless of emotion
    ("prosody", "semantic"): 0.1,   # Medium -- emphasis patterns relate to meaning
    ("prosody", "speaker"): 0.1,    # Medium -- each speaker has characteristic prosody
    ("prosody", "sentence"): 0.1,   # Medium -- sentence structure affects rhythm
    ("prosody", "voice"): 0.1,      # Medium -- voice quality affects prosodic perception
    ("semantic", "speaker"): 0.01,  # Weak -- topic is independent of who speaks
    ("semantic", "sentence"): 0.3,  # Strong -- sentence meaning IS semantic content
    ("semantic", "voice"): 0.01,    # Weak
    ("speaker", "sentence"): 0.01,  # Weak
    ("speaker", "voice"): 0.3,      # Strong -- same speaker = consistent voice
    ("sentence", "voice"): 0.01,    # Weak
}


def _get_physics_scale(a: str, b: str) -> float:
    """Look up the physics scale for a pair regardless of ordering."""
    if (a, b) in BRIDGE_PHYSICS:
        return BRIDGE_PHYSICS[(a, b)]
    if (b, a) in BRIDGE_PHYSICS:
        return BRIDGE_PHYSICS[(b, a)]
    return 0.01  # Default to weak


@dataclass
class CMBConfig:
    """Configuration for Cross-Modal Bridges."""
    semantic_dim: int = SEMANTIC_DIM
    use_bias: bool = True


class CrossModalBridge(nn.Module):
    """A single cross-modal bridge between two SPD output spaces.

    Linear projection that encodes the physical relationship between
    two modalities for a specific person. Small, fast, frozen after training.

    Args:
        name: Human-readable name for logging.
        init_scale: Physics-informed initialization scale.
        config: CMB configuration.
    """

    def __init__(
        self,
        name: str = "",
        init_scale: float = 0.1,
        config: CMBConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or CMBConfig()
        self.name = name
        self.proj = nn.Linear(cfg.semantic_dim, cfg.semantic_dim, bias=cfg.use_bias)
        # Initialize near-identity so bridge starts as pass-through
        nn.init.eye_(self.proj.weight)
        self.proj.weight.data *= init_scale
        if cfg.use_bias and self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project from source SPD space to target SPD space."""
        return torch.sigmoid(self.proj(x))

    def consistency_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute how well this bridge predicts the target from the source.

        Args:
            source: (B, 32) SPD output from source modality.
            target: (B, 32) SPD output from target modality.

        Returns:
            Scalar MSE loss.
        """
        predicted = self.forward(source)
        return F.mse_loss(predicted, target)


class CrossModalBridgeSet(nn.Module):
    """Complete set of cross-modal bridges for a person.

    21 bidirectional bridges between all 7 SPD spaces (C(7,2)=21 pairs):
      visual <-> emotion, visual <-> prosody, visual <-> semantic,
      visual <-> speaker, visual <-> sentence, visual <-> voice,
      emotion <-> prosody, emotion <-> semantic, emotion <-> speaker,
      emotion <-> sentence, emotion <-> voice,
      prosody <-> semantic, prosody <-> speaker, prosody <-> sentence,
      prosody <-> voice,
      semantic <-> speaker, semantic <-> sentence, semantic <-> voice,
      speaker <-> sentence, speaker <-> voice,
      sentence <-> voice
    """

    def __init__(self, config: CMBConfig | None = None) -> None:
        super().__init__()
        cfg = config or CMBConfig()

        bridges: dict[str, CrossModalBridge] = {}
        for a, b in combinations(MODALITY_NAMES, 2):
            scale = _get_physics_scale(a, b)
            bridges[f"{a}_to_{b}"] = CrossModalBridge(f"{a}->{b}", scale, cfg)
            bridges[f"{b}_to_{a}"] = CrossModalBridge(f"{b}->{a}", scale, cfg)

        self.bridges = nn.ModuleDict(bridges)

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "CrossModalBridgeSet: %d bridges, %d params", len(self.bridges), total,
        )

    def forward(
        self,
        spd_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply all bridges and return cross-modal predictions.

        Args:
            spd_outputs: Dict with keys from MODALITY_NAMES,
                         each (B, 32) SPD output tensor.

        Returns:
            Dict of bridge_name -> (B, 32) predicted target.
        """
        predictions = {}
        for name, bridge in self.bridges.items():
            src_name = name.split("_to_")[0]
            if src_name in spd_outputs:
                predictions[name] = bridge(spd_outputs[src_name])
        return predictions

    def total_consistency_loss(
        self,
        spd_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute total cross-modal consistency loss across all 21 pairs.

        For each bridge, the predicted target should match the actual
        target SPD output.

        Args:
            spd_outputs: Dict with keys from MODALITY_NAMES.

        Returns:
            Scalar loss (mean across all active bridges).
        """
        total = torch.tensor(0.0, device=next(iter(spd_outputs.values())).device)
        n = 0

        for name, bridge in self.bridges.items():
            src_name, tgt_name = name.split("_to_")
            if src_name in spd_outputs and tgt_name in spd_outputs:
                loss = bridge.consistency_loss(
                    spd_outputs[src_name],
                    spd_outputs[tgt_name],
                )
                total = total + loss
                n += 1

        return total / max(n, 1)

    def freeze(self) -> None:
        """Freeze all bridge weights. Called after training."""
        for p in self.parameters():
            p.requires_grad_(False)
        logger.info(
            "CrossModalBridgeSet: frozen (%d params)",
            sum(p.numel() for p in self.parameters()),
        )


# ---------------------------------------------------------------------------
# Santa-specific physics constants (measured from data)
# ---------------------------------------------------------------------------
@dataclass
class SantaPhysics:
    """Measured physical constants from Santa's interview data.

    These are used to initialize CMB weights and validate bridge outputs.
    """
    # Voice
    f0_mean: float = 137.0   # Hz
    f0_range: tuple[float, float] = (100.0, 200.0)
    f0_smile_shift: float = 12.0  # Hz increase when smiling

    # Emotion ranges (observed in data)
    arousal_range: tuple[float, float] = (0.15, 0.22)
    valence_range: tuple[float, float] = (0.504, 0.509)
    energy_range: tuple[float, float] = (0.296, 0.347)

    # Face
    jaw_range_px: tuple[int, int] = (365, 383)
    smile_energy_boost: float = 0.15

    # Speaking rate
    rate_range_wpm: tuple[float, float] = (50.0, 250.0)


def train_bridges(
    bridge_set: CrossModalBridgeSet,
    spd_visual: torch.Tensor,
    spd_emotion: torch.Tensor,
    spd_prosody: torch.Tensor,
    spd_semantic: torch.Tensor,
    spd_speaker: torch.Tensor,
    spd_sentence: torch.Tensor,
    spd_voice: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict[str, float]:
    """Train all cross-modal bridges on paired SPD outputs.

    Args:
        bridge_set: The CMB set to train.
        spd_visual: (N, 32) visual SPD outputs.
        spd_emotion: (N, 32) emotion SPD outputs.
        spd_prosody: (N, 32) prosody SPD outputs.
        spd_semantic: (N, 32) semantic SPD outputs.
        spd_speaker: (N, 32) speaker SPD outputs.
        spd_sentence: (N, 32) sentence SPD outputs.
        spd_voice: (N, 32) voice SPD outputs.
        epochs: Training epochs.
        lr: Learning rate.
        device: CUDA device.

    Returns:
        Dict of final loss values per bridge.
    """
    bridge_set = bridge_set.to(device)
    bridge_set.train()

    optimizer = torch.optim.Adam(bridge_set.parameters(), lr=lr)

    spd_outputs = {
        "visual": spd_visual.to(device),
        "emotion": spd_emotion.to(device),
        "prosody": spd_prosody.to(device),
        "semantic": spd_semantic.to(device),
        "speaker": spd_speaker.to(device),
        "sentence": spd_sentence.to(device),
        "voice": spd_voice.to(device),
    }

    best_loss = float("inf")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = bridge_set.total_consistency_loss(spd_outputs)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.info("CMB epoch %d/%d: loss=%.6f", epoch + 1, epochs, loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()

    # Compute per-bridge losses for reporting
    bridge_set.eval()
    losses = {}
    with torch.no_grad():
        for name, bridge in bridge_set.bridges.items():
            src_name, tgt_name = name.split("_to_")
            if src_name in spd_outputs and tgt_name in spd_outputs:
                l = bridge.consistency_loss(spd_outputs[src_name], spd_outputs[tgt_name])
                losses[name] = l.item()

    logger.info("CMB training done. Best loss: %.6f", best_loss)
    for name, val in sorted(losses.items()):
        logger.info("  %s: %.6f", name, val)

    return losses
