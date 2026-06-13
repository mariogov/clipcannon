"""Full State Verification for portable path resolution (GitHub issue #29).

Source of truth: the real filesystem. We build a real HF-cache-structured
directory tree and prove the resolver finds the snapshot via refs/main, removes
the hardcoded user path, and FAILS LOUD (no silent substitution, no download)
when a required artifact is missing.
"""
from __future__ import annotations

import pytest

from clipcannon.paths import hf_snapshot_dir, resolve_external_dir


def _make_repo(cache_root, repo_id, rev="deadbeefcafe", with_ref=True, n_snapshots=1):
    """Create a real models--org--name/{refs/main, snapshots/<rev>/config.json}."""
    repo = cache_root / ("models--" + repo_id.replace("/", "--"))
    snaps = repo / "snapshots"
    snaps.mkdir(parents=True)
    revs = [rev] if n_snapshots == 1 else [f"{rev}{i}" for i in range(n_snapshots)]
    for r in revs:
        (snaps / r).mkdir()
        (snaps / r / "config.json").write_text("{}")
    if with_ref:
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(revs[0])  # point at a real snapshot
    return repo, snaps / revs[0]


def test_resolves_snapshot_via_refs_main(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    _repo, expected = _make_repo(cache, "Acme/Model7B", rev="abc123", n_snapshots=3)
    monkeypatch.setenv("CLIPCANNON_MODELS_DIR", str(cache))

    resolved = hf_snapshot_dir("Acme/Model7B")
    print(f"[FSV] resolved -> {resolved}")
    assert resolved == expected
    assert resolved.is_dir()
    assert (resolved / "config.json").exists()  # the actual model files are here
    assert "/home/cabdru" not in str(resolved) or str(cache) in str(resolved)


def test_single_snapshot_without_ref(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    _repo, expected = _make_repo(cache, "Acme/Solo", rev="onlyone", with_ref=False, n_snapshots=1)
    monkeypatch.setenv("CLIPCANNON_MODELS_DIR", str(cache))
    assert hf_snapshot_dir("Acme/Solo") == expected


def test_missing_required_raises_with_searched_paths(tmp_path, monkeypatch):
    cache = tmp_path / "empty_cache"
    cache.mkdir()
    monkeypatch.setenv("CLIPCANNON_MODELS_DIR", str(cache))
    with pytest.raises(FileNotFoundError) as exc:
        hf_snapshot_dir("Acme/NotCached", required=True)
    msg = str(exc.value)
    print(f"[FSV] error -> {msg}")
    assert "Acme/NotCached" in msg
    assert "not pre-cached" in msg
    assert str(cache) in msg  # searched paths are reported


def test_missing_optional_returns_expected_path_no_raise(tmp_path, monkeypatch):
    cache = tmp_path / "empty_cache"
    cache.mkdir()
    monkeypatch.setenv("CLIPCANNON_MODELS_DIR", str(cache))
    p = hf_snapshot_dir("Acme/NotCached", required=False)
    print(f"[FSV] optional missing -> {p}")
    assert not p.exists()  # downstream isdir() check will report absence
    assert "models--Acme--NotCached" in str(p)


def test_resolve_external_dir_env_override(tmp_path, monkeypatch):
    install = tmp_path / "echomimic_v3"
    install.mkdir()
    (install / "infer_flash.py").write_text("# entrypoint")
    monkeypatch.setenv("CLIPCANNON_ECHOMIMIC_DIR", str(install))
    resolved = resolve_external_dir(
        "CLIPCANNON_ECHOMIMIC_DIR", tmp_path / "default", must_contain=("infer_flash.py",)
    )
    print(f"[FSV] external dir -> {resolved}")
    assert resolved == install


def test_resolve_external_dir_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPCANNON_ECHOMIMIC_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_external_dir("CLIPCANNON_ECHOMIMIC_DIR", tmp_path / "nope")


def test_resolve_external_dir_missing_required_entry_raises(tmp_path, monkeypatch):
    install = tmp_path / "wrong_dir"
    install.mkdir()  # exists but lacks infer_flash.py
    monkeypatch.setenv("CLIPCANNON_ECHOMIMIC_DIR", str(install))
    with pytest.raises(FileNotFoundError, match="missing required entry"):
        resolve_external_dir(
            "CLIPCANNON_ECHOMIMIC_DIR", tmp_path / "default", must_contain=("infer_flash.py",)
        )


def test_no_hardcoded_user_path_in_source():
    """The grep contract for #29: no functional /home/cabdru literals in src/."""
    import subprocess

    root = __import__("pathlib").Path(__file__).parent.parent / "src"
    out = subprocess.run(
        ["grep", "-rn", "/home/cabdru", str(root)], capture_output=True, text=True
    )
    print(f"[FSV] grep /home/cabdru in src -> {out.stdout.strip() or 'CLEAN'}")
    assert out.returncode != 0 or out.stdout.strip() == "", f"hardcoded paths remain:\n{out.stdout}"
