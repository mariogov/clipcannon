"""Voice Agent -- Personal AI Assistant."""

# Issue #59: enforce no-runtime-model-downloads before any ML import. Done inline
# (stdlib only) so voiceagent has no hard import dependency on clipcannon.
import os as _os

for _k, _v in (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"),
):
    _os.environ.setdefault(_k, _v)

__version__ = "0.1.0"
