"""Process-wide enforcement of *no runtime model downloads* (GitHub issue #59).

Root cause this fixes: model loaders were scattered across ``clipcannon``,
``phoenix`` and ``voiceagent``; some set ``HF_HUB_OFFLINE`` locally, many called
``from_pretrained(...)`` / ``hf_hub_download(...)`` with no offline guarantee, so
a missing model could silently trigger a slow, non-deterministic network
download (and fail in offline/air-gapped runs).

The fix is a single, import-time lever. ``huggingface_hub`` and ``transformers``
read ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` **once, at import**, so the
env vars MUST be set before those libraries are first imported. Importing this
module from each package's ``__init__`` (the very first code that runs for the
package) guarantees that ordering.

``setdefault`` is deliberate: the default is OFFLINE, but a dedicated
cache-warming step (``scripts/download_models.py``) may export
``HF_HUB_OFFLINE=0`` to fetch models on purpose. Normal runtime never does.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Env vars that disable implicit network access for the ML stack.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",          # huggingface_hub: no hub requests
    "TRANSFORMERS_OFFLINE": "1",    # transformers: local files only
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}


def enforce_offline() -> None:
    """Disable runtime model downloads for this process (idempotent).

    Uses ``setdefault`` so an explicit ``HF_HUB_OFFLINE=0`` (cache warming)
    is honoured, but the default for every normal entrypoint is offline.
    """
    for key, val in _OFFLINE_ENV.items():
        os.environ.setdefault(key, val)


def is_offline_enforced() -> bool:
    """True iff downloads are disabled (HF_HUB_OFFLINE truthy)."""
    return os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")


def require_cached_model(repo_or_path: str, *, search_roots: list[Path] | None = None) -> None:
    """Fail LOUD, *before* a loader call, when a model is not pre-cached.

    This gives an actionable error (what / where / why / how-to-fix) instead of
    the opaque ``LocalEntryNotFoundError`` huggingface raises in offline mode.

    Args:
        repo_or_path: an HF repo id (``org/name``) or a local directory path.
        search_roots: optional explicit cache roots to report in the error.
    """
    # Local directory form: just check it exists and is non-empty.
    p = Path(repo_or_path).expanduser()
    if p.is_dir():
        if any(p.iterdir()):
            return
        raise FileNotFoundError(
            f"Model directory is empty: {p}\n"
            f"  what:  required model files are missing at this path.\n"
            f"  why:   runtime downloads are disabled (HF_HUB_OFFLINE=1, issue #59).\n"
            f"  fix:   populate {p} via scripts/download_models.py (with network), "
            f"then re-run."
        )

    # HF repo id form: look for a models--org--name dir under known cache roots.
    if "/" in repo_or_path and not repo_or_path.startswith((".", "/")):
        roots = search_roots or _default_hf_roots()
        marker = "models--" + repo_or_path.replace("/", "--")
        searched = []
        for root in roots:
            cand = root / marker
            searched.append(str(cand))
            if cand.is_dir() and (cand / "snapshots").is_dir():
                snaps = [d for d in (cand / "snapshots").iterdir() if d.is_dir()]
                if snaps:
                    return
        raise FileNotFoundError(
            f"Model not pre-cached: {repo_or_path}\n"
            f"  what:  no local HF snapshot for this repo.\n"
            f"  where: searched {searched}\n"
            f"  why:   runtime downloads are disabled (HF_HUB_OFFLINE=1, issue #59).\n"
            f"  fix:   run `HF_HUB_OFFLINE=0 huggingface-cli download {repo_or_path}` "
            f"(or scripts/download_models.py), then re-run offline."
        )
    # Unknown form — let the caller's own check handle it.


def _default_hf_roots() -> list[Path]:
    roots: list[Path] = []
    for env in ("CLIPCANNON_MODELS_DIR", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            roots.append(Path(v).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


# Enforce immediately on import so any later `import transformers` is offline.
enforce_offline()
