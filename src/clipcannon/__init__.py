"""ClipCannon: AI-powered video understanding and editing pipeline via MCP."""

# Issue #59: disable runtime model downloads process-wide BEFORE any ML library
# (transformers / huggingface_hub) can be imported. Importing this first is the
# whole point — the env vars are read once, at those libraries' import time.
from clipcannon import offline as _offline  # noqa: F401

_offline.enforce_offline()

__version__ = "0.1.0"
