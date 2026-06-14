"""Portable path resolution — no hardcoded user paths, no runtime downloads.

Replaces hardcoded absolute user paths and brittle pinned
HuggingFace snapshot hashes with resolution that:

* honours environment overrides first,
* searches the standard + ClipCannon model-cache roots,
* reads the HF cache's ``refs/main`` to find the current snapshot (so the
  commit hash is not pinned in source),
* and FAILS LOUD with the searched locations when a required artifact is
  missing — never downloads at runtime, never silently substitutes.
"""
from __future__ import annotations

import os
from pathlib import Path


def hf_cache_roots() -> list[Path]:
    """Candidate HuggingFace-style cache roots, in priority order.

    A "root" is a directory that contains ``models--{org}--{name}`` folders.
    """
    roots: list[Path] = []
    # Explicit overrides win.
    for env in ("CLIPCANNON_MODELS_DIR", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            roots.append(Path(v).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    home = Path.home()
    roots.extend(
        [
            home / ".cache" / "huggingface" / "hub",
            home / ".clipcannon" / "models",
            home / ".clipcannon" / "models" / "qwen3-8b-hf",
        ]
    )
    # De-dup, preserve order.
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _resolve_snapshot(repo_root: Path) -> Path | None:
    """Given a ``models--{org}--{name}`` dir, return its current snapshot dir.

    Reads ``refs/main`` to pick the snapshot (de-brittles the hash). Falls back
    to the sole snapshot if there is exactly one. Returns None if unresolvable.
    """
    snapshots = repo_root / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = repo_root / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8").strip()
        cand = snapshots / rev
        if cand.is_dir():
            return cand
    subdirs = [d for d in snapshots.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return None


def hf_snapshot_dir(repo_id: str, *, required: bool = True) -> Path:
    """Resolve the local snapshot directory for a cached HF repo.

    Args:
        repo_id: e.g. ``"Qwen/Qwen3-8B"``.
        required: if True, raise when the model is not cached; if False, return
            the expected standard-cache path (which will not exist) so callers'
            own ``isdir`` checks can report the absence in context.

    Raises:
        FileNotFoundError: when ``required`` and the model is not pre-cached.
    """
    repo_folder = "models--" + repo_id.replace("/", "--")
    searched: list[Path] = []
    for root in hf_cache_roots():
        repo_root = root / repo_folder
        searched.append(repo_root)
        snap = _resolve_snapshot(repo_root)
        if snap is not None:
            return snap
    if required:
        raise FileNotFoundError(
            f"Model '{repo_id}' is not pre-cached. Searched: "
            f"{', '.join(str(s) for s in searched)}. "
            f"Pre-download it (e.g. `huggingface-cli download {repo_id}`) or set "
            f"CLIPCANNON_MODELS_DIR / HF_HOME. Runtime downloads are disabled by design."
        )
    # Non-raising: return the standard expected path so downstream isdir() fails
    # naturally and reports this path.
    return (Path.home() / ".cache" / "huggingface" / "hub" / repo_folder / "snapshots")


def resolve_external_dir(
    env_var: str,
    default: Path,
    *,
    must_contain: tuple[str, ...] = (),
) -> Path:
    """Resolve an external install directory from an env var (or default).

    Args:
        env_var: environment variable holding an override path.
        default: fallback path (typically under ``Path.home()``).
        must_contain: required relative entries that prove this is the right dir.

    Raises:
        FileNotFoundError: if the directory or a required entry is missing.
    """
    raw = os.environ.get(env_var)
    d = Path(raw).expanduser() if raw else default
    if not d.is_dir():
        raise FileNotFoundError(
            f"{env_var} resolves to '{d}', which does not exist. "
            f"Set {env_var} to the installation directory."
        )
    for entry in must_contain:
        if not (d / entry).exists():
            raise FileNotFoundError(
                f"'{d}' is missing required entry '{entry}' — this does not look "
                f"like the expected installation. Set {env_var} correctly."
            )
    return d


def resolve_ml_python() -> str:
    """Resolve a torch-capable Python interpreter for ML subprocesses.

    The MCP server runs in a lightweight venv WITHOUT torch; ML subprocesses
    (e.g. the wake listener's Silero VAD) need the torch-capable interpreter.
    Resolution order (portable, no hardcoded user path):

      1. ``CLIPCANNON_ML_PYTHON`` env override,
      2. the running interpreter if it already has torch,
      3. common conda/mamba base interpreters,
      4. fall back to ``sys.executable`` (caller should verify torch).
    """
    import shutil
    import sys

    env = os.environ.get("CLIPCANNON_ML_PYTHON")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return str(p)

    if _interpreter_has_torch(sys.executable):
        return sys.executable

    home = Path.home()
    candidates = [
        home / "miniconda3" / "bin" / "python3",
        home / "anaconda3" / "bin" / "python3",
        home / "miniforge3" / "bin" / "python3",
        home / "mambaforge" / "bin" / "python3",
    ]
    conda_exe = shutil.which("python3")
    if conda_exe:
        candidates.append(Path(conda_exe))
    for c in candidates:
        if c.exists() and _interpreter_has_torch(str(c)):
            return str(c)

    return sys.executable


def _interpreter_has_torch(python_exe: str) -> bool:
    """True iff `python_exe` can import torch (cached per-process)."""
    import subprocess

    cache = _interpreter_has_torch.__dict__.setdefault("_cache", {})
    if python_exe in cache:
        return cache[python_exe]
    try:
        rc = subprocess.run(
            [python_exe, "-c", "import torch"],
            capture_output=True, timeout=60,
        ).returncode
        ok = rc == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    cache[python_exe] = ok
    return ok
