"""Full State Verification for packaging / MCP wiring (GitHub issues #1, #2, #61).

These are real-state checks, not mocks:

* #1  — `uv lock` must actually resolve the whole dependency graph (base + every
        extra) and the committed lockfile must be current; pyproject must carry
        the license-table fix and the override-dependencies that make it solvable.
* #2  — the tracked .mcp.json must declare the clipcannon MCP server.
* #61 — the configured launch command must really start the server and register
        tools (subprocess against the real binary), and must contain no hardcoded
        per-user absolute path (that would regress #29).

Source of truth: the real files on disk (pyproject.toml, uv.lock, .mcp.json,
scripts/clipcannon-mcp) and the real subprocess behaviour of the launcher.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (REPO / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


# --------------------------------------------------------------------------- #
# #1 — dependency resolution
# --------------------------------------------------------------------------- #
def test_license_is_freeform_table_not_invalid_spdx():
    """BSL-1.1 is not valid SPDX; it must be a license table so the build works."""
    proj = _pyproject()["project"]
    lic = proj["license"]
    print(f"[FSV] project.license = {lic!r}")
    assert isinstance(lic, dict), "license must be a table {text=...}, not an SPDX string"
    assert lic.get("text") == "BSL-1.1"


def test_uv_overrides_relax_the_known_bad_pins():
    """The overrides that make the graph solvable must be present (issue #1)."""
    overrides = _pyproject().get("tool", {}).get("uv", {}).get("override-dependencies", [])
    print(f"[FSV] override-dependencies = {overrides}")
    joined = " ".join(overrides).lower()
    # torch pin from audiocraft==2.1.0 and av 11.x are the two hard blockers.
    assert "torch>=" in joined.replace(" ", ""), "torch override missing"
    assert "av>=14" in joined.replace(" ", ""), "av>=14 override missing"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_uv_lock_is_current_and_resolvable():
    """`uv lock --locked` proves the committed lockfile resolves the full graph.

    --locked fails non-zero if the lockfile is missing, stale, or unsatisfiable,
    so this is real-state verification of resolution, not a stored boolean.
    """
    proc = subprocess.run(
        ["uv", "lock", "--locked", "--offline"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(f"[FSV] uv lock --locked rc={proc.returncode}\n{proc.stderr[-800:]}")
    assert proc.returncode == 0, f"lockfile not current/resolvable: {proc.stderr[-400:]}"


# --------------------------------------------------------------------------- #
# #2 / #61 — MCP server wiring + launch
# --------------------------------------------------------------------------- #
def test_mcp_json_declares_clipcannon_server():
    cfg = json.loads((REPO / ".mcp.json").read_text())
    servers = cfg.get("mcpServers", {})
    print(f"[FSV] .mcp.json servers = {list(servers)}")
    assert "clipcannon" in servers, "clipcannon server missing from .mcp.json (#2)"


def test_mcp_json_has_no_hardcoded_user_path():
    """A committed config must not bake in /home/<user>/... (regresses #29)."""
    raw = (REPO / ".mcp.json").read_text()
    assert not re.search(r"/home/[^/]+/", raw), f"hardcoded user path in .mcp.json: {raw}"


def test_launcher_exists_and_is_executable():
    launcher = REPO / "scripts" / "clipcannon-mcp"
    assert launcher.exists(), "launcher script missing (#61)"
    assert os.access(launcher, os.X_OK), "launcher not executable"
    body = launcher.read_text()
    assert not re.search(r"/home/[^/]+/", body), "launcher hardcodes a user path"


def test_launcher_actually_starts_server_and_registers_tools():
    """Subprocess the real launcher; it must boot the MCP server and log tools.

    Source of truth: the launcher's own stderr/stdout from a real process. We run
    it from /tmp (arbitrary cwd) to prove it self-resolves the repo + venv.
    """
    launcher = REPO / "scripts" / "clipcannon-mcp"
    proc = subprocess.run(
        [str(launcher)],
        cwd="/tmp",
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"(\d+) tools registered", out)
    print(f"[FSV] launcher tools-registered match = {m.group(0) if m else None}")
    assert m is not None, f"server did not register tools; output:\n{out[-1500:]}"
    assert int(m.group(1)) > 0


def test_launcher_fails_loud_when_venv_missing(tmp_path):
    """Edge case: no venv -> must exit non-zero with an actionable diagnostic."""
    fake = tmp_path / "scripts"
    fake.mkdir()
    shutil.copy(REPO / "scripts" / "clipcannon-mcp", fake / "clipcannon-mcp")
    os.chmod(fake / "clipcannon-mcp", 0o755)
    proc = subprocess.run(
        [str(fake / "clipcannon-mcp")],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(f"[FSV] missing-venv rc={proc.returncode}\n{proc.stderr}")
    assert proc.returncode != 0
    assert "uv sync" in proc.stderr  # tells the user exactly how to fix it
