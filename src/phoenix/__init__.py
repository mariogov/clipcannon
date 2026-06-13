"""Phoenix — GPU-native avatar engine for ClipCannon.

This package provides zero-copy GPU compositing, 3D Gaussian Splat
rendering, and embedding-driven avatar behavior. All operations
run entirely on GPU via CuPy and CUDA kernels.

Exports:
    Error hierarchy: PhoenixError, CompositorError, RenderError,
        ExpressionError, BehaviorError
    Configuration: PhoenixConfig, CompositorConfig, RenderConfig,
        BehaviorWeights
    Expression engine: EmotionFusion, EmotionState, ProsodyFeatures,
        SpeakerTracker, SpeakerInfo, GestureLibrary, GestureClip
"""

# Issue #59: enforce no-runtime-model-downloads before any ML import. Inline
# (stdlib only) to avoid a hard import dependency on clipcannon.
import os as _os  # noqa: E402

for _k, _v in (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"),
):
    _os.environ.setdefault(_k, _v)

from phoenix.config import (  # noqa: E402
    BehaviorWeights,
    CompositorConfig,
    PhoenixConfig,
    RenderConfig,
)
from phoenix.errors import (
    BehaviorError,
    CompositorError,
    ExpressionError,
    PhoenixError,
    RenderError,
)
from phoenix.expression import (
    EmotionFusion,
    EmotionState,
    GestureClip,
    GestureLibrary,
    ProsodyFeatures,
    SpeakerInfo,
    SpeakerTracker,
)

__all__ = [
    "BehaviorError",
    "BehaviorWeights",
    "CompositorConfig",
    "CompositorError",
    "EmotionFusion",
    "EmotionState",
    "ExpressionError",
    "GestureClip",
    "GestureLibrary",
    "PhoenixConfig",
    "PhoenixError",
    "ProsodyFeatures",
    "RenderConfig",
    "RenderError",
    "SpeakerInfo",
    "SpeakerTracker",
]
