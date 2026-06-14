"""GPU precision auto-detection based on CUDA compute capability.

This selects the *native compute dtype* used to load models — i.e. the precision
the hardware runs efficiently and that maps to a real ``torch.dtype``:

    - Blackwell  (CC 12.0): bf16  (5th-gen tensor cores; bf16 is the safe default)
    - Hopper     (CC 9.0):  bf16
    - Ampere/Ada (CC 8.x):  bf16  (bf16 supported since CC 8.0)
    - Turing     (CC 7.5):  fp16  (no bf16 tensor-core support)
    - Volta      (CC 7.0):  fp16
    - No GPU / unsupported: fp32  (CPU fallback)

NOTE: NVFP4 / INT8 are *quantization* formats applied on top of a model, not
native load dtypes, and they need dedicated kernels (TensorRT / FP4). They are
selected by the separate quantization path (issues #43/#44), NOT here — returning
"nvfp4" from this function previously misreported the precision (no consumer
could map it to a torch.dtype) and contradicted the actual bf16/fp16 loads.
"""

from __future__ import annotations

import logging

from clipcannon.exceptions import GPUError

logger = logging.getLogger(__name__)

# CC string -> native compute dtype. bf16 for Ampere+ (CC >= 8.0), fp16 for
# Turing/Volta. At runtime we additionally trust torch.cuda.is_bf16_supported()
# as the authoritative check (see auto_detect_precision).
PRECISION_MAP: dict[str, str] = {
    "12.0": "bf16",  # Blackwell (RTX 5090, sm_120)
    "9.0": "bf16",   # Hopper
    "8.9": "bf16",   # Ada Lovelace
    "8.6": "bf16",   # Ampere (consumer)
    "8.0": "bf16",   # Ampere (A100)
    "7.5": "fp16",   # Turing
    "7.0": "fp16",   # Volta
}

DEFAULT_CPU_PRECISION = "fp32"

#: native precision string -> torch dtype (so the value is actually usable).
_PRECISION_TO_TORCH = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def precision_to_torch_dtype(precision: str):  # noqa: ANN201 - returns torch.dtype
    """Map a native precision string to a real ``torch.dtype``.

    Raises GPUError for non-native (quantization) strings so a caller never
    silently mis-loads a model in a format that is not a load dtype.
    """
    import torch

    name = _PRECISION_TO_TORCH.get(precision)
    if name is None:
        raise GPUError(
            f"'{precision}' is not a native load dtype. Valid: "
            f"{sorted(_PRECISION_TO_TORCH)}. Quantization formats (nvfp4/int8) are "
            f"applied via the quantization path (#43/#44), not as a load dtype."
        )
    return getattr(torch, name)


def _is_torch_available() -> bool:
    """Check whether PyTorch is importable.

    Returns:
        True if torch can be imported, False otherwise.
    """
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _is_cuda_available() -> bool:
    """Check whether CUDA is available via PyTorch.

    Returns:
        True if CUDA is available, False otherwise.
    """
    if not _is_torch_available():
        return False
    import torch

    return torch.cuda.is_available()


def get_compute_capability(device_index: int = 0) -> str | None:
    """Get the CUDA compute capability string for a device.

    Args:
        device_index: CUDA device index (default 0).

    Returns:
        Compute capability as "major.minor" string, or None if no GPU.
    """
    if not _is_cuda_available():
        return None

    try:
        import torch

        major, minor = torch.cuda.get_device_capability(device_index)
        cc = f"{major}.{minor}"
        logger.info("Detected GPU compute capability: %s", cc)
        return cc
    except Exception as exc:
        logger.warning("Failed to get compute capability for device %d: %s", device_index, exc)
        return None


def auto_detect_precision(device_index: int = 0) -> str:
    """Auto-detect the optimal precision for the current GPU.

    Looks up the GPU compute capability in PRECISION_MAP. Falls back
    to fp32 (CPU mode) if no GPU is detected or the compute capability
    is not in the map.

    Args:
        device_index: CUDA device index (default 0).

    Returns:
        Native precision string: "bf16", "fp16", or "fp32".
    """
    cc = get_compute_capability(device_index)

    if cc is None:
        logger.warning(
            "No CUDA GPU detected. Using CPU mode with %s precision.", DEFAULT_CPU_PRECISION
        )
        return DEFAULT_CPU_PRECISION

    # Authoritative runtime check: if the device + driver report bf16 tensor-core
    # support, bf16 is the safe native default (Ampere/Hopper/Blackwell). This is
    # more robust than a static CC map and forward-compatible with new archs.
    try:
        import torch

        if torch.cuda.is_bf16_supported():
            logger.info("Selected precision: bf16 (CC %s, bf16 supported)", cc)
            return "bf16"
    except Exception as exc:  # noqa: BLE001 - fall through to the static map
        logger.debug("bf16 support probe failed (%s); using CC map", exc)

    precision = PRECISION_MAP.get(cc)
    if precision is None:
        # Prefix-match for forward compatibility, then a numeric fallback.
        major = cc.split(".")[0]
        for map_cc, map_precision in sorted(PRECISION_MAP.items(), reverse=True):
            if map_cc.startswith(major + "."):
                precision = map_precision
                break

    if precision is None:
        cc_float = float(cc)
        if cc_float >= 8.0:
            precision = "bf16"  # Ampere+ supports bf16
        elif cc_float >= 7.0:
            precision = "fp16"
        else:
            precision = DEFAULT_CPU_PRECISION
        logger.warning("CC %s not in map. Falling back to %s precision.", cc, precision)

    logger.info("Selected precision: %s (CC %s)", precision, cc)
    return precision


def validate_gpu_for_pipeline(device_index: int = 0) -> dict[str, str | int | float | bool]:
    """Validate that the GPU meets minimum requirements for the pipeline.

    Args:
        device_index: CUDA device index (default 0).

    Returns:
        Dictionary with GPU validation results.

    Raises:
        GPUError: If torch is available but CUDA reports errors.
    """
    result: dict[str, str | int | float | bool] = {
        "torch_available": _is_torch_available(),
        "cuda_available": _is_cuda_available(),
        "cpu_only": True,
    }

    if not _is_torch_available():
        result["warning"] = "PyTorch not installed. Running in CPU-only mode."
        return result

    if not _is_cuda_available():
        result["warning"] = "No CUDA GPU detected. Running in CPU-only mode."
        return result

    try:
        import torch

        props = torch.cuda.get_device_properties(device_index)
        vram_gb = props.total_memory / (1024**3)

        result.update(
            {
                "cpu_only": False,
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_total_gb": round(vram_gb, 2),
                "precision": auto_detect_precision(device_index),
                "meets_minimum": vram_gb >= 8.0,
            }
        )

        if vram_gb < 8.0:
            result["warning"] = (
                f"GPU {props.name} has only {vram_gb:.1f} GB VRAM. "
                "Minimum 8 GB recommended for sequential pipeline execution."
            )

    except Exception as exc:
        raise GPUError(
            f"Failed to validate GPU device {device_index}: {exc}",
            details={"device_index": device_index},
        ) from exc

    return result
