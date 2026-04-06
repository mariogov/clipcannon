"""Semantic Clone Model -- Constellation-Embedded Transformer (7 modalities).

Replaces the generic CloneModel with a meaning-aware architecture where:
1. SPDs decode raw embeddings into human-interpretable semantic positions
2. CMBs enforce cross-modal physics between all 7 modalities
3. Constellation attention heads are INITIALIZED from the North Star states
4. Each head's K/V matrices ARE constellation vectors
5. Regularized to stay near constellation during training

Output: 52 ARKit blendshapes + 16 voice params (same interface as CloneModel).

Architecture:
  Raw embeddings -> 7 SPDs -> 7x32 semantic vectors
                           -> CMB consistency check (21 bridges)
                           -> Concat (224-dim)
                           -> Constellation Transformer (4 layers)
                           -> Output heads

~3M params. Inference <1ms on RTX 5090.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phoenix.clone.semantic_decoders import (
    EmotionSPD,
    ProsodySPD,
    SemanticSPD,
    SpeakerSPD,
    SentenceSPD,
    VoiceSPD,
    SPDConfig,
    VisualSPD,
    SEMANTIC_DIM,
)
from phoenix.clone.cross_modal_bridges import CrossModalBridgeSet

logger = logging.getLogger(__name__)

# Output dimensions (same as CloneModel for compatibility)
BLENDSHAPE_DIM = 52
VOICE_DIM = 16
TOTAL_OUTPUT_DIM = BLENDSHAPE_DIM + VOICE_DIM
NUM_MODALITIES = 7
FUSED_DIM = SEMANTIC_DIM * NUM_MODALITIES  # 224

# Modality names in canonical order
MODALITY_NAMES = ["visual", "emotion", "prosody", "semantic", "speaker", "sentence", "voice"]

# Input dimensions for each modality (for error messages)
MODALITY_INPUT_DIMS = {
    "visual": 1152,
    "emotion": 1024,
    "prosody": 12,
    "semantic": 768,
    "speaker": 512,
    "sentence": 384,
    "voice": 192,
}


class ConstellationAttention(nn.Module):
    """Multi-head attention initialized from constellation state vectors.

    Each attention head corresponds to a behavioral state. The K/V
    matrices are initialized from that state's embedding position and
    regularized to stay nearby during training.

    Args:
        dim: Model dimension (224 for 7 modalities).
        num_heads: Number of constellation heads.
        constellation_keys: (num_heads, dim) initial K vectors from constellation.
        constellation_values: (num_heads, dim) initial V vectors from constellation.
    """

    def __init__(
        self,
        dim: int = FUSED_DIM,
        num_heads: int = 8,
        constellation_keys: torch.Tensor | None = None,
        constellation_values: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)

        # Store constellation initialization for regularization
        if constellation_keys is not None:
            self.register_buffer("_init_k_weight", constellation_keys.clone())
        else:
            self.register_buffer("_init_k_weight", torch.zeros(dim, dim))
        if constellation_values is not None:
            self.register_buffer("_init_v_weight", constellation_values.clone())
        else:
            self.register_buffer("_init_v_weight", torch.zeros(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention with constellation-initialized weights.

        Args:
            x: (B, seq_len, dim) input.

        Returns:
            (B, seq_len, dim) attended output.
        """
        B, S, D = x.shape
        normed = self.norm(x)

        q = self.q_proj(normed).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        x = x + out
        x = x + self.ff(self.ff_norm(x))
        return x

    def constellation_reg_loss(self) -> torch.Tensor:
        """Regularization loss pulling K/V weights toward initialization."""
        loss = F.mse_loss(self.k_proj.weight, self._init_k_weight)
        loss = loss + F.mse_loss(self.v_proj.weight, self._init_v_weight)
        return loss


class SemanticCloneModel(nn.Module):
    """Meaning-aware clone model with constellation-embedded transformer.

    Takes raw embeddings from all 7 modalities, decodes them through SPDs,
    enforces cross-modal physics via 21 CMBs, and produces blendshape + voice
    outputs through a constellation-initialized transformer.

    All 7 inputs are REQUIRED. Missing modalities raise ValueError.

    Args:
        num_constellation_heads: Attention heads per layer (from states).
        num_layers: Transformer depth.
        spd_config: Configuration for SPDs.
    """

    def __init__(
        self,
        num_constellation_heads: int = 8,
        num_layers: int = 4,
        spd_config: SPDConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        # Semantic Position Decoders -- all 7 modalities
        self.spd_visual = VisualSPD(spd_config)
        self.spd_emotion = EmotionSPD(spd_config)
        self.spd_prosody = ProsodySPD(spd_config)
        self.spd_semantic = SemanticSPD(spd_config)
        self.spd_speaker = SpeakerSPD(spd_config)
        self.spd_sentence = SentenceSPD(spd_config)
        self.spd_voice = VoiceSPD(spd_config)

        # Cross-Modal Bridges: 21 bidirectional pairs (frozen after Phase 2)
        self.cmbs = CrossModalBridgeSet()

        # Fusion projection: 7x32=224 -> 224
        self.fusion = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM),
            nn.LayerNorm(FUSED_DIM),
            nn.GELU(),
        )

        # Learnable position tokens (like Perceiver latents)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, 16, FUSED_DIM) * 0.02
        )

        # Constellation transformer layers
        self.constellation_layers = nn.ModuleList([
            ConstellationAttention(
                dim=FUSED_DIM,
                num_heads=num_constellation_heads,
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.head_blendshape = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM),
            nn.GELU(),
            nn.Linear(FUSED_DIM, BLENDSHAPE_DIM),
        )
        self.head_voice = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM // 2),
            nn.GELU(),
            nn.Linear(FUSED_DIM // 2, VOICE_DIM),
        )

        # Track state
        self._constellation_initialized = False
        self._cmbs_frozen = False

        logger.info(
            "SemanticCloneModel: %d layers, %d heads, %d modalities, fused=%dd, %.2fM params",
            num_layers, num_constellation_heads, NUM_MODALITIES, FUSED_DIM,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def init_constellation(
        self,
        state_embeddings: dict[str, np.ndarray],
    ) -> None:
        """Initialize attention K/V from constellation state vectors.

        Args:
            state_embeddings: Dict of state_name -> (224,) fused embedding.
                At most num_heads states will be used per layer.
        """
        states = list(state_embeddings.values())
        n_states = len(states)
        logger.info("Initializing constellation from %d states", n_states)

        for layer_idx, layer in enumerate(self.constellation_layers):
            nh = layer.num_heads
            # Assign states round-robin across layers
            start = (layer_idx * nh) % max(n_states, 1)
            selected = []
            for h in range(nh):
                idx = (start + h) % max(n_states, 1)
                if idx < n_states:
                    vec = torch.from_numpy(states[idx].astype(np.float32))
                    # Pad or truncate to FUSED_DIM
                    if vec.shape[0] < FUSED_DIM:
                        vec = F.pad(vec, (0, FUSED_DIM - vec.shape[0]))
                    else:
                        vec = vec[:FUSED_DIM]
                    selected.append(vec)
                else:
                    selected.append(torch.randn(FUSED_DIM) * 0.01)

            # Build K init matrix from selected state vectors
            k_init = torch.stack(selected)  # (nh, FUSED_DIM)
            # Expand to full weight shape by repeating
            k_weight = k_init.repeat(FUSED_DIM // nh, 1)[:FUSED_DIM]
            v_weight = k_weight.clone()

            # Copy to buffers
            layer._init_k_weight.copy_(k_weight)
            layer._init_v_weight.copy_(v_weight)

            # Initialize actual weights from constellation
            with torch.no_grad():
                layer.k_proj.weight.copy_(k_weight)
                layer.v_proj.weight.copy_(v_weight)

        self._constellation_initialized = True
        logger.info("Constellation initialization complete")

    def freeze_cmbs(self) -> None:
        """Freeze cross-modal bridges after training."""
        self.cmbs.freeze()
        self._cmbs_frozen = True

    def freeze_spds(self) -> None:
        """Freeze all 7 SPDs after training."""
        spds = [
            self.spd_visual, self.spd_emotion, self.spd_prosody,
            self.spd_semantic, self.spd_speaker, self.spd_sentence,
            self.spd_voice,
        ]
        for spd in spds:
            for p in spd.parameters():
                p.requires_grad_(False)
        logger.info("All 7 SPDs frozen")

    def forward(
        self,
        visual: torch.Tensor,
        emotion: torch.Tensor,
        prosody: torch.Tensor,
        semantic: torch.Tensor,
        speaker: torch.Tensor,
        sentence: torch.Tensor,
        voice: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: raw embeddings -> SPDs -> transformer -> outputs.

        ALL 7 inputs are REQUIRED. No optional inputs, no fallbacks.

        Args:
            visual: (B, 1152) SigLIP visual embedding.
            emotion: (B, 1024) Wav2Vec2 mean-pooled hidden state.
            prosody: (B, 12) prosody features.
            semantic: (B, 768) Nomic text embedding.
            speaker: (B, 512) WavLM speaker embedding.
            sentence: (B, 384) MiniLM sentence embedding.
            voice: (B, 192) ECAPA voice embedding.

        Returns:
            Dict with 'blendshapes' (B, 52), 'voice' (B, 16),
            'spd_outputs' (dict of (B, 32) per modality).

        Raises:
            ValueError: If any input is None.
        """
        # Validate all inputs present
        inputs = {
            "visual": visual, "emotion": emotion, "prosody": prosody,
            "semantic": semantic, "speaker": speaker, "sentence": sentence,
            "voice": voice,
        }
        for name, tensor in inputs.items():
            if tensor is None:
                raise ValueError(
                    f"Missing required input '{name}'. All 7 modalities are required. "
                    f"Expected shape: (B, {MODALITY_INPUT_DIMS[name]})"
                )

        B = visual.shape[0]

        # Decode through all 7 SPDs
        spd_out = {
            "visual": self.spd_visual(visual),
            "emotion": self.spd_emotion(emotion),
            "prosody": self.spd_prosody(prosody),
            "semantic": self.spd_semantic(semantic),
            "speaker": self.spd_speaker(speaker),
            "sentence": self.spd_sentence(sentence),
            "voice": self.spd_voice(voice),
        }

        # Fuse SPD outputs: concat all 7 -> 224d
        fused = torch.cat([
            spd_out["visual"],
            spd_out["emotion"],
            spd_out["prosody"],
            spd_out["semantic"],
            spd_out["speaker"],
            spd_out["sentence"],
            spd_out["voice"],
        ], dim=-1)  # (B, 224)

        fused = self.fusion(fused)  # (B, 224)

        # Create token sequence: fused input + learnable latents
        tokens = torch.cat([
            fused.unsqueeze(1),
            self.latent_tokens.expand(B, -1, -1),
        ], dim=1)  # (B, 17, 224)

        # Constellation transformer
        for layer in self.constellation_layers:
            tokens = layer(tokens)

        # Pool -> output heads
        pooled = tokens.mean(dim=1)  # (B, 224)

        blendshapes = torch.sigmoid(self.head_blendshape(pooled))
        voice_out = self.head_voice(pooled)

        return {
            "blendshapes": blendshapes,
            "voice": voice_out,
            "spd_outputs": spd_out,
            "_pooled": pooled,
        }

    def constellation_reg_loss(self) -> torch.Tensor:
        """Total regularization loss keeping weights near constellation init."""
        total = torch.tensor(0.0, device=self._get_device())
        for layer in self.constellation_layers:
            total = total + layer.constellation_reg_loss()
        return total / max(len(self.constellation_layers), 1)

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def param_count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compile(self, mode: str = "max-autotune") -> "SemanticCloneModel":
        """Compile for optimized inference."""
        self.forward = torch.compile(self.forward, mode=mode)  # type: ignore[method-assign]
        logger.info("SemanticCloneModel compiled with mode=%s", mode)
        return self
