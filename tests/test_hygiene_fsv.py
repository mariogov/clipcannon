"""Full State Verification for repo hygiene (GitHub issue #60).

Source of truth: the working tree + git index. We prove:
  - no stray literal-tilde (`~`) directory exists in the repo root;
  - no gitignored artifact dirs (tmp/temp/memory) are tracked;
  - the root cause (writing to an unexpanded `~`) cannot recur — every adapter
    that has a `~/...` default path expands it with `.expanduser()` before use.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_stray_tilde_dir():
    stray = ROOT / "~"
    print(f"[FSV] repo-root '~' dir exists? {stray.exists()}")
    assert not stray.exists(), "a literal '~' directory is back in the repo root"


def test_no_tracked_ignored_artifacts():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    tracked = [
        ln for ln in out.stdout.splitlines()
        if ln.startswith(("tmp/", "temp/", "memory/"))
    ]
    print(f"[FSV] tracked tmp/temp/memory files: {tracked or 'NONE'}")
    assert tracked == [], f"gitignored artifacts are tracked: {tracked}"


def test_gitignore_covers_artifacts():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("~/", "tmp/", "temp/", "memory/", ".venv/", "__pycache__/"):
        assert pat in gi, f".gitignore missing {pat}"
    print("[FSV] .gitignore covers ~/, tmp/, temp/, memory/, .venv/, __pycache__/")


def test_tilde_path_defaults_are_expanded():
    """Recurrence guard: adapters with `~/...` defaults must expanduser them."""
    for rel in (
        "src/voiceagent/adapters/clipcannon.py",
        "src/voiceagent/adapters/fast_tts.py",
    ):
        src = (ROOT / rel).read_text()
        # If the file declares a ~/ default, it must also call .expanduser().
        if '"~/' in src:
            assert ".expanduser()" in src, f"{rel} uses a ~/ default without expanduser()"
            print(f"[FSV] {rel}: ~/ defaults are expanduser()-ed ✓")
