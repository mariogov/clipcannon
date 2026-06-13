"""Full State Verification for doc/count reconciliation (GitHub issue #62).

Source of truth: `scripts/count_artifacts.py` (introspects the live tool
registry + AST-counts the pipeline stages). README and whitepaper MUST match it.
This test fails if the docs drift from the code again.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_counts() -> dict:
    spec = importlib.util.spec_from_file_location(
        "count_artifacts", ROOT / "scripts" / "count_artifacts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()


def test_readme_matches_introspected_counts():
    counts = _load_counts()
    readme = (ROOT / "README.md").read_text()
    print(f"[FSV] authoritative counts: {counts}")

    # Tool count: badge + prose must equal the introspected total.
    assert f"MCP_tools-{counts['tools']}" in readme, "README tools badge != introspected count"
    assert f"**{counts['tools']} tools**" in readme
    # No stale numbers.
    assert "MCP_tools-54" not in readme
    assert "**54 tools**" not in readme

    # Stage count.
    assert f"{counts['stages']}-stage" in readme
    assert "22-stage" not in readme and "22 stages" not in readme


def test_readme_tool_table_sums_to_total():
    counts = _load_counts()
    readme = (ROOT / "README.md").read_text()
    # Sum the "| **Category** | N |" rows in the MCP tools table.
    nums = [int(m) for m in re.findall(r"^\| \*\*[^|]+\*\* \| (\d+) \|", readme, re.M)]
    print(f"[FSV] README tool-table counts: {nums} sum={sum(nums)} vs total={counts['tools']}")
    assert nums, "no tool-table rows parsed"
    assert sum(nums) == counts["tools"], f"table sums to {sum(nums)}, not {counts['tools']}"


def test_whitepaper_stage_count():
    counts = _load_counts()
    wp = (ROOT / "docs/clipcannon_whitepaper.md").read_text()
    print(f"[FSV] whitepaper checked for {counts['stages']}-stage")
    assert "22-stage" not in wp and "22-Stage" not in wp
    assert f"{counts['stages']}-stage" in wp or f"{counts['stages']}-Stage" in wp
