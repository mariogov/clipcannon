"""Backward-compat shim: make transformers 5.x `check_model_inputs` accept the
transformers-4.x decorator-factory call style used by qwen_tts (issue #35/#32/#33).

Root cause: ``qwen_tts`` (pins transformers==4.57.x) decorates its forward() with
``@check_model_inputs()`` — in 4.x that was a decorator *factory* (call with
parens). In transformers 5.x ``check_model_inputs(func)`` became a *direct*
decorator with ``func`` required, so ``check_model_inputs()`` raises
``TypeError: missing 1 required positional argument: 'func'`` at qwen_tts import.

The ClipCannon analysis pipeline depends on transformers 5.x, so we can't
downgrade. This shim wraps ``transformers.utils.generic.check_model_inputs`` so
BOTH styles work — ``@check_model_inputs`` (5.x, func passed) and
``@check_model_inputs(...)`` (4.x factory) — and must be applied BEFORE qwen_tts
is imported (qwen_tts does ``from transformers.utils.generic import
check_model_inputs`` at its module top).
"""
from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_check_model_inputs() -> None:
    """Idempotently make check_model_inputs accept both 4.x and 5.x call styles."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from transformers.utils import generic as _generic
    except ImportError:
        return

    orig = getattr(_generic, "check_model_inputs", None)
    if orig is None:
        return

    # If the original already supports a no-arg factory call, nothing to do.
    try:
        params = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        return
    required = [
        p for p in params.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if not required:
        # already factory-style (4.x) — no patch needed
        _PATCHED = True
        return

    def _compat(func=None, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        # 5.x bare usage: @check_model_inputs  -> func is the decorated callable
        if callable(func) and not args and not kwargs:
            return orig(func)
        # 4.x factory usage: @check_model_inputs(...) -> return a decorator that
        # forwards just the callable (5.x's check_model_inputs ignores the old
        # config kwargs, which were advisory).
        def _decorator(f):  # noqa: ANN001, ANN202
            return orig(f)
        return _decorator

    _compat.__wrapped__ = orig
    _generic.check_model_inputs = _compat
    _PATCHED = True
    logger.info("Applied transformers 5.x check_model_inputs compat shim for qwen_tts")


# Legacy token-id attributes transformers 4.x exposed on every PretrainedConfig
# (defaulting to None). transformers 5.x removed them (they live on the
# generation config now), so qwen_tts's `config.pad_token_id` etc. raise
# AttributeError. Restore the 4.x behaviour: missing -> None.
_LEGACY_TOKEN_ATTRS = frozenset({
    "pad_token_id", "bos_token_id", "eos_token_id",
    "sep_token_id", "cls_token_id", "mask_token_id", "decoder_start_token_id",
})
_CONFIG_PATCHED = False


def patch_legacy_config_token_ids() -> None:
    """Make missing legacy *_token_id config attrs return None (4.x behaviour)."""
    global _CONFIG_PATCHED
    if _CONFIG_PATCHED:
        return
    try:
        from transformers.configuration_utils import PretrainedConfig
    except ImportError:
        return
    orig_getattribute = PretrainedConfig.__getattribute__

    def _getattribute(self, name):  # noqa: ANN001, ANN202
        try:
            return orig_getattribute(self, name)
        except AttributeError:
            if name in _LEGACY_TOKEN_ATTRS:
                return None
            raise

    PretrainedConfig.__getattribute__ = _getattribute
    _CONFIG_PATCHED = True
    logger.info("Applied transformers 5.x legacy token-id config compat shim")


_ROPE_PATCHED = False


def patch_default_rope() -> None:
    """Re-register the standard ('default') RoPE init that transformers 5.x
    removed from ROPE_INIT_FUNCTIONS (qwen_tts configs use rope_type='default').

    This is the canonical, stable default-rotary formula (unchanged across
    transformers versions): inv_freq = 1/base**(arange(0,dim,2)/dim).
    """
    global _ROPE_PATCHED
    if _ROPE_PATCHED:
        return
    try:
        from transformers import modeling_rope_utils as _r
    except ImportError:
        return
    funcs = getattr(_r, "ROPE_INIT_FUNCTIONS", None)
    if funcs is None or "default" in funcs:
        _ROPE_PATCHED = True
        return

    import torch

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):  # noqa: ANN001, ANN003, ANN202, ARG001
        base = getattr(config, "rope_theta", 10000.0)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0  # (inv_freq, attention_scaling)

    funcs["default"] = _compute_default_rope_parameters
    _ROPE_PATCHED = True
    logger.info("Registered 'default' RoPE init for transformers 5.x (qwen_tts)")


_STATIC_LAYER_PATCHED = False


def patch_static_layer_lazy_init() -> None:
    """faster_qwen3_tts's CUDA-graph capture calls StaticLayer.lazy_initialization
    with only key_states (transformers 4.x signature). transformers 5.x requires
    (key_states, value_states). Default value_states to key_states — only the
    shape/dtype/device matter for the lazy cache allocation, and k/v are identical
    there, so this is faithful."""
    global _STATIC_LAYER_PATCHED
    if _STATIC_LAYER_PATCHED:
        return
    try:
        from transformers.cache_utils import StaticLayer
    except ImportError:
        return
    orig = StaticLayer.lazy_initialization
    try:
        params = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        return
    if "value_states" not in params:
        _STATIC_LAYER_PATCHED = True
        return

    def _lazy_init(self, key_states, value_states=None):  # noqa: ANN001, ANN202
        if value_states is None:
            value_states = key_states
        return orig(self, key_states, value_states)

    StaticLayer.lazy_initialization = _lazy_init
    _STATIC_LAYER_PATCHED = True
    logger.info("Applied StaticLayer.lazy_initialization compat shim (transformers 5.x)")


def patch_all() -> None:
    """Apply every transformers-5.x compat shim qwen_tts needs."""
    patch_check_model_inputs()
    patch_legacy_config_token_ids()
    patch_default_rope()
    patch_static_layer_lazy_init()
