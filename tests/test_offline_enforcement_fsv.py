"""FSV for no-runtime-model-downloads enforcement (GitHub issue #59).

Source of truth: the *actual* process environment after importing each package,
and the *actual* exception raised by a real loader when a model is not cached.
No mocks — we attempt a real (uncached) HF load and prove it refuses to hit the
network instead of asserting on a stored flag.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _env_after_import(pkg: str) -> dict[str, str]:
    """Import `pkg` in a clean subprocess, dump the offline env vars it set."""
    code = (
        "import os, json, sys;"
        f"sys.path.insert(0, {str(SRC)!r});"
        # Make sure we observe what the package import sets, not a pre-set env.
        "[os.environ.pop(k, None) for k in "
        "('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY')];"
        f"__import__({pkg!r});"
        "print(json.dumps({k: os.environ.get(k) for k in "
        "('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY')}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, f"import {pkg} failed: {out.stderr[-1500:]}"
    import json

    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("pkg", ["clipcannon", "voiceagent", "phoenix"])
def test_importing_package_enforces_offline(pkg):
    env = _env_after_import(pkg)
    print(f"[FSV] after `import {pkg}` -> {env}")
    assert env["HF_HUB_OFFLINE"] == "1", f"{pkg} did not enforce HF_HUB_OFFLINE"
    assert env["TRANSFORMERS_OFFLINE"] == "1", f"{pkg} did not enforce TRANSFORMERS_OFFLINE"


def test_require_cached_model_raises_actionably_for_uncached_repo():
    from clipcannon.offline import require_cached_model

    fake = "clipcannon-test/definitely-not-cached-zzz999"
    with pytest.raises(FileNotFoundError) as ei:
        require_cached_model(fake)
    msg = str(ei.value)
    print(f"[FSV] require_cached_model error:\n{msg}")
    # Actionable: names the repo, says why, and how to fix.
    assert fake in msg
    assert "HF_HUB_OFFLINE" in msg
    assert "fix:" in msg


def test_require_cached_model_passes_for_real_cached_dir(tmp_path):
    from clipcannon.offline import require_cached_model

    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    require_cached_model(str(d))  # must NOT raise — directory is non-empty


def test_empty_model_dir_fails_loud(tmp_path):
    from clipcannon.offline import require_cached_model

    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError) as ei:
        require_cached_model(str(d))
    assert "empty" in str(ei.value).lower()


@pytest.mark.integration
def test_real_transformers_load_of_uncached_repo_refuses_network():
    """The decisive check: a real `from_pretrained` of an uncached repo must
    raise an *offline* error fast, never download. Requires transformers."""
    import time

    transformers = pytest.importorskip("transformers")
    # Importing clipcannon first guarantees the offline env is set.
    import clipcannon  # noqa: F401

    t0 = time.time()
    with pytest.raises(Exception) as ei:  # noqa: BLE001 - HF raises a specific subclass
        transformers.AutoConfig.from_pretrained("clipcannon-test/nope-not-real-zzz")
    dt = time.time() - t0
    err = f"{type(ei.value).__name__}: {ei.value}"
    print(f"[FSV] offline from_pretrained raised in {dt:.2f}s -> {err}")
    # Offline failures are local-cache lookups: must be fast (no network retry).
    assert dt < 10, "took too long — likely attempted a network download"
    assert "offline" in err.lower() or "local" in err.lower() or "couldn't" in err.lower()
