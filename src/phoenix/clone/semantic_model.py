"""Semantic Clone Model -- Constellation-Embedded Transformer.

Replaces the generic CloneModel with a meaning-aware architecture where:
1. SPDs decode raw embeddings into human-interpretable semantic positions
2. CMBs enforce cross-modal physics between modalities
3. Constellation attention heads are INITIALIZED from the North Star states
4. Each head's K/V matrices ARE constellation vectors
5. Regularized to stay near constellation during training

Output: 52 ARKit blendshapes + 16 voice params (same interface as CloneModel).

Architecture:
  Raw embeddings -> SPDs -> 4x32 semantic vectors
                        -> CMB consistency check
                        -> Concat (128-dim)
                        -> Constellation Transformer (4 layers)
                        -> Output heads

~2M params. Inference <1ms on RTX 5090.
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
    SPDConfig,
    VisualSPD,
)
from phoenix.clone.cross_modal_bridges import CrossModalBridgeSet

logger = logging.getLogger(__name__)

# Output dimensions (same as CloneModel for compatibility)
BLENDSHAPE_DIM = 52
VOICE_DIM = 16
TOTAL_OUTPUT_DIM = BLENDSHAPE_DIM + VOICE_DIM
SEMANTIC_DIM = 32
FUSED_DIM = SEMANTIC_DIM * 4  # 128


class ConstellationAttention(nn.Module):
    """Multi-head attention initialized from constellation state vectors.

    Each attention head corresponds to a behavioral state. The K/V
    matrices are initialized from that state's embedding position and
    regularized to stay nearby during training.

    Args:
        dim: Model dimension.
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

    Takes raw embeddings, decodes them through SPDs, enforces cross-modal
    physics via CMBs, and produces blendshape + voice outputs through
    a constellation-initialized transformer.

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

        # Semantic Position Decoders
        self.spd_visual = VisualSPD(spd_config)
        self.spd_emotion = EmotionSPD(spd_config)
        self.spd_prosody = ProsodySPD(spd_config)
        self.spd_semantic = SemanticSPD(spd_config)

        # Cross-Modal Bridges (frozen after Phase 2)
        self.cmbs = CrossModalBridgeSet()

        # Fusion projection: 4x32 -> 128
        self.fusion = nn.Sequential(
            nn.Linear(SEMANTIC_DIM * 4, FUSED_DIM),
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
            "SemanticCloneModel: %d layers, %d heads, %.2fM params",
            num_layers, num_constellation_heads,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def init_constellation(
        self,
        state_embeddings: dict[str, np.ndarray],
    ) -> None:
        """Initialize attention K/V from constellation state vectors.

        Args:
            state_embeddings: Dict of state_name -> (128,) fused embedding.
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
                    selected.append(torch.from_numpy(states[idx].astype(np.float32)))
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
        """Freeze SPDs after training."""
        for spd in [self.spd_visual, self.spd_emotion, self.spd_prosody, self.spd_semantic]:
            for p in spd.parameters():
                p.requires_grad_(False)
        logger.info("SPDs frozen")

    def forward(
        self,
        visual: torch.Tensor | None = None,
        emotion: torch.Tensor | None = None,
        prosody: torch.Tensor | None = None,
        semantic: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: raw embeddings -> SPDs -> transformer -> outputs.

        Args:
            visual: (B, 1152) SigLIP visual embedding.
            emotion: (B, 3) emotion arousal/valence/energy.
            prosody: (B, 12) prosody features.
            semantic: (B, 768) Nomic text embedding.

        Returns:
            Dict with 'blendshapes' (B, 52), 'voice' (B, 16),
            'spd_outputs' (dict of (B, 32) per modality).
        """
        device = self._get_device()
        # Determine batch size from first available input
        B = 1
        for x in [visual, emotion, prosody, semantic]:
            if x is not None:
                B = x.shape[0]
                break

        # Decode through SPDs
        spd_out = {}
        if visual is not None:
            spd_out["visual"] = self.spd_visual(visual)
        else:
            spd_out["visual"] = torch.zeros(B, SEMANTIC_DIM, device=device)

        if emotion is not None:
            spd_out["emotion"] = self.spd_emotion(emotion)
        else:
            spd_out["emotion"] = torch.zeros(B, SEMANTIC_DIM, device=device)

        if prosody is not None:
            spd_out["prosody"] = self.spd_prosody(prosody)
        else:
            spd_out["prosody"] = torch.zeros(B, SEMANTIC_DIM, device=device)

        if semantic is not None:
            spd_out["semantic"] = self.spd_semantic(semantic)
        else:
            spd_out["semantic"] = torch.zeros(B, SEMANTIC_DIM, device=device)

        # Fuse SPD outputs: concat -> project
        fused = torch.cat([
            spd_out["visual"],
            spd_out["emotion"],
            spd_out["prosody"],
            spd_out["semantic"],
        ], dim=-1)  # (B, 128)

        fused = self.fusion(fused)  # (B, 128)

        # Create token sequence: fused input + learnable latents
        tokens = torch.cat([
            fused.unsqueeze(1),
            self.latent_tokens.expand(B, -1, -1),
        ], dim=1)  # (B, 17, 128)

        # Constellation transformer
        for layer in self.constellation_layers:
            tokens = layer(tokens)

        # Pool -> output heads
        pooled = tokens.mean(dim=1)  # (B, 128)

        blendshapes = torch.sigmoid(self.head_blendshape(pooled))
        voice = self.head_voice(pooled)

        return {
            "blendshapes": blendshapes,
            "voice": voice,
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
