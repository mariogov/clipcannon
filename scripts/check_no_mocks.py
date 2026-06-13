#!/usr/bin/env python3
"""CI gate: forbid mocks + broken-state-masking in tests (GitHub issue #58).

Why: nothing structurally prevented a future test from mocking the DB/model or
passing while the system is broken. This gate institutionalises the no-mock /
no-cover-up rule so it cannot silently regress.

What it flags (fails CI, exit 1):
  * any use of unittest.mock / MagicMock / AsyncMock / @patch / pytest-mock,
    EXCEPT in files on the explicit, justified allowlist below;
  * obvious broken-state-masking patterns: `assert True`, bare `except: pass`
    in a test body, unconditional `pytest.skip()`/`pytest.xfail()` with no
    condition/reason.

The allowlist only permits patches that redirect *our own* path / DB-location /
env seams to REAL temporary data (dependency injection). Mocking model or DB
*behaviour* is never allowed.

Run:  python scripts/check_no_mocks.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"

# file (relative to tests/) -> reason it may use unittest.mock.patch.
# These patch path/DB-location/env seams only; real sqlite/real logic runs.
ALLOWLIST: dict[str, str] = {
    "test_billing.py": (
        "patches license-server DB_DIR/DB_PATH to a real temp sqlite db and "
        "clears os.environ — path/env seam, real crypto + real db underneath."
    ),
    "fsv_pipeline_tools.py": (
        "patches _get_projects_dir/_get_db_path to a real temp project dir — "
        "path seam; the MCP tools run against real sqlite databases."
    ),
}

MOCK_PATTERNS = [
    re.compile(r"\bunittest\.mock\b"),
    re.compile(r"\bfrom\s+unittest\s+import\s+mock\b"),
    re.compile(r"\bimport\s+mock\b"),
    re.compile(r"\bMagicMock\b"),
    re.compile(r"\bAsyncMock\b"),
    re.compile(r"\bmock\.patch\b"),
    re.compile(r"@patch\b"),
    re.compile(r"\bpytest_mock\b"),
]


def _scan_mocks(path: Path) -> list[str]:
    hits: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pat in MOCK_PATTERNS:
            if pat.search(line):
                hits.append(f"{path.relative_to(REPO)}:{i}: {stripped}")
                break
    return hits


def _scan_masking(path: Path) -> list[str]:
    """AST-based detection of obvious broken-state-masking."""
    hits: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:  # a test that doesn't even parse is itself broken
        return [f"{path.relative_to(REPO)}: SyntaxError {e}"]

    for node in ast.walk(tree):
        # `assert True` / `assert 1` — a no-op assertion that can never fail.
        if isinstance(node, ast.Assert):
            t = node.test
            if (isinstance(t, ast.Constant) and bool(t.value)) or (
                isinstance(t, ast.Constant) and t.value == 1
            ):
                hits.append(f"{path.relative_to(REPO)}:{node.lineno}: `assert True` no-op")
        # The real masking signal: a try whose body contains an *assertion* and a
        # handler that just `pass`es — i.e. a failing assert is silently swallowed
        # so the test passes while the system is broken. Benign cleanup/parse/
        # retry try/except (no assert in the body) is intentionally NOT flagged.
        if isinstance(node, ast.Try):
            body_has_assert = any(
                isinstance(n, ast.Assert)
                for stmt in node.body
                for n in ast.walk(stmt)
            )
            for h in node.handlers:
                only_pass = len(h.body) == 1 and isinstance(h.body[0], ast.Pass)
                if only_pass and body_has_assert:
                    hits.append(
                        f"{path.relative_to(REPO)}:{h.lineno}: "
                        f"`except: pass` swallows an assertion in the try body"
                    )
        # unconditional skip/xfail at statement level (no marker/condition).
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            is_pytest_skip = (
                node.func.attr in ("skip", "xfail")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pytest"
            )
            if is_pytest_skip:
                # allow skip with a reason kwarg/arg (conditional skips are fine)
                has_reason = bool(node.args) or any(k.arg == "reason" for k in node.keywords)
                if not has_reason:
                    hits.append(
                        f"{path.relative_to(REPO)}:{node.lineno}: "
                        f"unconditional pytest.{node.func.attr}() with no reason"
                    )
    return hits


def main() -> int:
    test_files = sorted(p for p in TESTS.rglob("*.py") if "__pycache__" not in p.parts)
    mock_violations: list[str] = []
    masking_violations: list[str] = []
    allow_used: set[str] = set()

    for f in test_files:
        rel = f.name
        hits = _scan_mocks(f)
        if hits:
            if rel in ALLOWLIST:
                allow_used.add(rel)
            else:
                mock_violations.extend(hits)
        masking_violations.extend(_scan_masking(f))

    print(f"Scanned {len(test_files)} test files under {TESTS}.")
    if allow_used:
        print("\nAllowlisted mock usage (justified seams):")
        for rel in sorted(allow_used):
            print(f"  - {rel}: {ALLOWLIST[rel]}")

    ok = True
    if mock_violations:
        ok = False
        print("\n❌ FORBIDDEN MOCK USAGE (issue #58 — use real fixtures/fakes):")
        for v in mock_violations:
            print(f"  {v}")
    if masking_violations:
        ok = False
        print("\n❌ BROKEN-STATE-MASKING PATTERNS:")
        for v in masking_violations:
            print(f"  {v}")

    # Stale allowlist hygiene: an entry that no longer has any mock usage should
    # be removed so the allowlist stays honest.
    stale = set(ALLOWLIST) - allow_used
    if stale:
        print("\n⚠️  Stale allowlist entries (no mock usage found — remove them):")
        for rel in sorted(stale):
            print(f"  - {rel}")
        ok = False

    print("\n" + ("✅ PASS: no forbidden mocks or masking." if ok else "❌ FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
