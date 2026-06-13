"""Full State Verification for provenance chain integrity (GitHub issue #54).

Source of truth: the `provenance` table (SHA-256 hash chain) in a REAL project
DB. We record a real chain, then INDEPENDENTLY read the table back and prove:
  - an intact chain verifies (verified=True, every chain_hash present);
  - any tamper (content field, chain_hash, deletion) is detected and pinpointed.

Also asserts the half-migration is reconciled: provenance MCP tool definitions
are empty (internal-only) and the dashboard calls the internal API directly.

No mocks. No fallbacks.
"""
from __future__ import annotations

import sqlite3

from tests import fsv_harness as fsv

from clipcannon.db.connection import get_connection
from clipcannon.db.schema import create_project_db
from clipcannon.provenance import (
    ExecutionInfo,
    InputInfo,
    ModelInfo,
    OutputInfo,
    record_provenance,
    verify_chain,
)

PROJECT_ID = "proj_fsv_prov"


def _seed_project(db_path) -> None:
    conn = get_connection(db_path, enable_vec=False, dict_rows=False)
    try:
        conn.execute(
            "INSERT INTO project (project_id, name, source_path, source_sha256, "
            "duration_ms, resolution, fps, codec) VALUES (?,?,?,?,?,?,?,?)",
            (PROJECT_ID, "prov", "/tmp/p.mp4", "0" * 64, 1000, "1920x1080", 30.0, "h264"),
        )
        conn.commit()
    finally:
        conn.close()


def _record(db_path, op, in_sha, out_sha, parent):
    return record_provenance(
        db_path=db_path,
        project_id=PROJECT_ID,
        operation=op,
        stage=op,
        input_info=InputInfo(file_path=f"/tmp/{op}.in", sha256=in_sha, size_bytes=10),
        output_info=OutputInfo(sha256=out_sha, record_count=1),
        model_info=ModelInfo(name=f"{op}-model", version="1.0"),
        execution_info=ExecutionInfo(duration_ms=5),
        parent_record_id=parent,
        description=f"{op} step",
    )


def _build_chain(db_path) -> list[str]:
    """Record a real 3-entry chain: genesis -> r2 -> r3."""
    r1 = _record(db_path, "probe", "a" * 64, "b" * 64, None)
    r2 = _record(db_path, "transcribe", "b" * 64, "c" * 64, r1)
    r3 = _record(db_path, "highlights", "c" * 64, "d" * 64, r2)
    return [r1, r2, r3]


def test_intact_chain_verifies(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db)
    ids = _build_chain(db)

    # FSV: read the source of truth and show the actual chain hashes.
    fsv.assert_rowcount(db, "provenance", 3)
    rows = fsv.raw_query(
        db, "SELECT record_id, parent_record_id, substr(chain_hash,1,12) AS h "
        "FROM provenance ORDER BY timestamp_utc, record_id"
    )
    for r in rows:
        fsv.evidence("provenance", f"{r['record_id']} parent={r['parent_record_id']} hash={r['h']}…")

    result = verify_chain(PROJECT_ID, db)
    fsv.evidence("verify_chain", f"verified={result.verified} total={result.total_records} broken_at={result.broken_at}")
    assert result.verified is True
    assert result.total_records == 3
    assert result.broken_at is None
    # every record has a non-empty chain_hash
    assert all(len(fsv.raw_query(db, "SELECT chain_hash FROM provenance WHERE record_id=?", (i,))[0]["chain_hash"]) == 64 for i in ids)


def test_tampered_content_field_detected(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db)
    ids = _build_chain(db)

    # EDGE 1: tamper a content field (input_sha256) of the MIDDLE record.
    raw = sqlite3.connect(str(db))
    raw.execute("UPDATE provenance SET input_sha256=? WHERE record_id=?", ("f" * 64, ids[1]))
    raw.commit()
    raw.close()

    result = verify_chain(PROJECT_ID, db)
    fsv.evidence("after content tamper", f"verified={result.verified} broken_at={result.broken_at} issue={result.issue}")
    assert result.verified is False
    assert result.broken_at == ids[1]


def test_tampered_chain_hash_detected(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db)
    ids = _build_chain(db)

    # EDGE 2: directly overwrite a stored chain_hash with a plausible-looking value.
    raw = sqlite3.connect(str(db))
    raw.execute("UPDATE provenance SET chain_hash=? WHERE record_id=?", ("0" * 64, ids[2]))
    raw.commit()
    raw.close()

    result = verify_chain(PROJECT_ID, db)
    fsv.evidence("after chain_hash tamper", f"verified={result.verified} broken_at={result.broken_at}")
    assert result.verified is False
    assert result.broken_at == ids[2]


def test_deleted_record_breaks_chain(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db)
    ids = _build_chain(db)

    # EDGE 3: delete the middle record — the child's parent link now dangles.
    raw = sqlite3.connect(str(db))
    raw.execute("DELETE FROM provenance WHERE record_id=?", (ids[1],))
    raw.commit()
    raw.close()

    fsv.assert_rowcount(db, "provenance", 2)
    result = verify_chain(PROJECT_ID, db)
    fsv.evidence("after deletion", f"verified={result.verified} broken_at={result.broken_at} issue={result.issue}")
    assert result.verified is False


def test_empty_project_chain(tmp_path):
    db = create_project_db(PROJECT_ID, base_dir=tmp_path)
    _seed_project(db)
    fsv.assert_rowcount(db, "provenance", 0)
    result = verify_chain(PROJECT_ID, db)
    fsv.evidence("empty chain", f"verified={result.verified} total={result.total_records}")
    # An empty chain is trivially intact (nothing has been tampered).
    assert result.total_records == 0
    assert result.verified is True


def test_provenance_is_internal_only_and_dashboard_reconciled():
    """Reconciliation: MCP defs empty; dashboard uses the internal API."""
    from clipcannon.tools.provenance_tools import PROVENANCE_TOOL_DEFINITIONS

    fsv.evidence("PROVENANCE_TOOL_DEFINITIONS", PROVENANCE_TOOL_DEFINITIONS)
    assert PROVENANCE_TOOL_DEFINITIONS == [], "provenance must stay internal-only (no MCP tools)"

    # Dashboard route imports the internal provenance API, not the removed tools.
    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "src/clipcannon/dashboard/routes/provenance.py"
    ).read_text()
    assert "from clipcannon.provenance import" in src
    assert "provenance_tools" not in src, "dashboard must not call removed MCP provenance tools"
    assert "dispatch_provenance_tool" not in src
