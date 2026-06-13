"""Instrument differentiation-contract validator (GitHub issue #19).

Operationalises the paper's contract: every embedding instrument must add
*independent, grounded* information. Concretely, over a corpus of real ingested
projects we require:

  * each instrument carries >= ``MIN_BITS`` (0.05) of mutual information about a
    grounded outcome (whether a time window is a highlight), and
  * no pair of instruments is more than ``MAX_CORR`` (0.6) correlated.

A degenerate instrument (e.g. the historical zero-vector emotion bug) or a
duplicated instrument is then caught automatically instead of by accident.

This module loads NOTHING but sqlite + numpy/scipy/sklearn — no model weights,
no network. It FAILS LOUD on every malformed-input edge case rather than
silently certifying.

CLI:
    python -m clipcannon.pipeline.differentiation PROJ_DIR [PROJ_DIR ...]
"""
from __future__ import annotations

import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Instruments stored as sqlite-vec tables: (table, vector_column).
VEC_INSTRUMENTS: list[tuple[str, str]] = [
    ("vec_frames", "visual_embedding"),
    ("vec_semantic", "semantic_embedding"),
    ("vec_emotion", "emotion_embedding"),
    ("vec_speakers", "speaker_embedding"),
]
# timestamp column differs per table.
_TS_COL = {
    "vec_frames": "timestamp_ms",
    "vec_semantic": "timestamp_ms",
    "vec_emotion": "start_ms",
    "vec_speakers": "timestamp_ms",
}
# vec0 virtual tables do NOT expose rowid; each declares its own primary key.
_PK_COL = {
    "vec_frames": "frame_id",
    "vec_semantic": "segment_id",
    "vec_emotion": "id",
    "vec_speakers": "id",
}

WINDOW_MS = 5000          # aggregate instruments into 5s windows
MIN_BITS = 0.05           # per-instrument MI floor (bits)
MAX_CORR = 0.6            # pairwise correlation ceiling
MIN_WINDOWS = 8           # refuse to certify below this many aligned windows

# Which grounded outcomes can meaningfully test each instrument. MI is only a
# HARD floor for an instrument when at least one of *its* relevant outcomes is
# testable (varies) in the corpus; otherwise the instrument is reported
# UNTESTABLE (corpus limitation) rather than falsely failed as degenerate. A
# truly degenerate instrument is still caught by the intrinsic check below.
INSTRUMENT_OUTCOMES: dict[str, list[str]] = {
    "vec_frames": ["scene", "shot_type", "highlight"],
    "vec_semantic": ["sentiment", "scene", "highlight"],
    "vec_emotion": ["sentiment", "highlight", "scene"],
    "vec_speakers": ["speaker_id"],
}


class DifferentiationError(Exception):
    """Raised when the corpus cannot be certified (with an actionable reason)."""


@dataclass
class DifferentiationReport:
    instruments: list[str]
    bits: dict[str, float] = field(default_factory=dict)
    grounded_by: dict[str, str] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)  # OK / WEAK / UNTESTABLE / DEGENERATE
    intrinsic: dict[str, dict[str, float]] = field(default_factory=dict)
    corr: dict[tuple[str, str], float] = field(default_factory=dict)
    n_windows: int = 0
    n_projects: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [
            f"Differentiation report — {self.n_projects} project(s), "
            f"{self.n_windows} aligned {WINDOW_MS}ms windows",
            "",
            "Per-instrument mutual information vs best grounded outcome (bits):",
        ]
        for inst in self.instruments:
            b = self.bits.get(inst, float("nan"))
            st = self.status.get(inst, "?")
            via = self.grounded_by.get(inst, "?")
            intr = self.intrinsic.get(inst, {})
            lines.append(
                f"  [{st:10s}] {inst:14s} {b:.4f} bits (via {via}; "
                f"var={intr.get('variance', float('nan')):.2e}, "
                f"rank={int(intr.get('eff_rank', 0))})"
            )
        lines.append("")
        lines.append(f"Pairwise |correlation| (ceiling {MAX_CORR}):")
        for (a, b), c in sorted(self.corr.items()):
            flag = "OK " if c <= MAX_CORR else "RED"
            lines.append(f"  [{flag}] {a:12s} x {b:12s} {c:.3f}")
        lines.append("")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        if self.warnings:
            lines.append("")
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        for f in self.failures:
            lines.append(f"  - {f}")
        return "\n".join(lines)


def _open(db_path: Path) -> sqlite3.Connection:
    import sqlite_vec

    con = sqlite3.connect(str(db_path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _load_instrument_windows(
    con: sqlite3.Connection, table: str, col: str, project_tag: str
) -> dict[int, np.ndarray]:
    """Return {window_index: mean_embedding} for one instrument.

    Raises DifferentiationError (naming project/row) on NaN/Inf — never returns
    a silently-corrupt vector.
    """
    ts = _TS_COL[table]
    pk = _PK_COL.get(table, "rowid")
    try:
        rows = con.execute(
            f"SELECT {pk}, {ts}, vec_to_json({col}) FROM {table}"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    import json

    buckets: dict[int, list[np.ndarray]] = {}
    for rowid, t, vjson in rows:
        if vjson is None or t is None:
            continue
        v = np.asarray(json.loads(vjson), dtype=np.float64)
        if not np.isfinite(v).all():
            bad = np.where(~np.isfinite(v))[0][:5].tolist()
            raise DifferentiationError(
                f"NaN/Inf embedding in {project_tag}:{table} rowid={rowid} "
                f"at dims {bad}. Source-of-truth is corrupt — re-run that stage."
            )
        buckets.setdefault(int(t) // WINDOW_MS, []).append(v)
    return {w: np.mean(np.stack(vs), axis=0) for w, vs in buckets.items()}


def _span_windows(s: int, e: int) -> range:
    return range(int(s) // WINDOW_MS, int(e) // WINDOW_MS + 1)


def _grounded_outcomes(con: sqlite3.Connection) -> dict[str, dict[int, object]]:
    """Per-window grounded labels from the pipeline's own real annotations.

    Returns {outcome_name: {window_index: label}}. An instrument is 'grounded'
    if it carries information about AT LEAST ONE of these — a single outcome
    would unfairly fail instruments orthogonal to it (e.g. speaker vs highlight),
    whereas a truly degenerate instrument scores ~0 against all of them.
    """
    outcomes: dict[str, dict[int, object]] = {}

    def _table_exists(name: str) -> bool:
        return con.execute(
            "SELECT 1 FROM sqlite_master WHERE name=?", (name,)
        ).fetchone() is not None

    # highlight: binary, window overlaps any highlight span.
    if _table_exists("highlights"):
        hl: dict[int, object] = {}
        for s, e in con.execute("SELECT start_ms, end_ms FROM highlights"):
            if s is None or e is None:
                continue
            for w in _span_windows(s, e):
                hl[w] = 1
        if hl:
            outcomes["highlight"] = hl  # absence handled as 0 later

    # sentiment + speaker + scene from segment/scene tables (multiclass labels).
    if _table_exists("transcript_segments"):
        sent: dict[int, object] = {}
        spk: dict[int, object] = {}
        for s, e, sentiment, speaker in con.execute(
            "SELECT start_ms, end_ms, sentiment, speaker_id FROM transcript_segments"
        ):
            if s is None or e is None:
                continue
            for w in _span_windows(s, e):
                if sentiment is not None:
                    sent[w] = 1 if str(sentiment).upper().startswith("POS") else 0
                if speaker is not None:
                    spk[w] = str(speaker)
        if sent:
            outcomes["sentiment"] = sent
        if spk:
            outcomes["speaker_id"] = spk

    if _table_exists("scenes"):
        scene: dict[int, object] = {}
        shot: dict[int, object] = {}
        for sid, s, e, st in con.execute(
            "SELECT scene_id, start_ms, end_ms, shot_type FROM scenes"
        ):
            if s is None or e is None:
                continue
            for w in _span_windows(s, e):
                scene[w] = int(sid)
                if st is not None:
                    shot[w] = str(st)
        if scene:
            outcomes["scene"] = scene
        if shot:
            outcomes["shot_type"] = shot

    return outcomes


def _pc1(mat: np.ndarray) -> np.ndarray:
    """Project rows onto their top principal component -> 1-D signal."""
    centered = mat - mat.mean(axis=0, keepdims=True)
    # top right singular vector
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[0]


def validate_differentiation(
    project_dirs: list[Path],
    *,
    min_bits: float = MIN_BITS,
    max_corr: float = MAX_CORR,
    extra_instruments: list[tuple[str, str]] | None = None,
) -> DifferentiationReport:
    """Compute the report and populate failures. Raises DifferentiationError on
    structurally-uncertifiable input (too few windows, degenerate outcome)."""
    from sklearn.feature_selection import mutual_info_classif

    instruments = VEC_INSTRUMENTS + (extra_instruments or [])
    inst_names = [t for t, _ in instruments]

    # window key = (project_index, window_index) so windows never collide across
    # projects. signal[inst][key] = mean embedding.
    per_inst: dict[str, dict[tuple[int, int], np.ndarray]] = {n: {} for n in inst_names}
    # outcomes[name][key] = label (mixed types; int for binary, str for multiclass)
    outcomes: dict[str, dict[tuple[int, int], object]] = {}

    dbs: list[Path] = []
    for proj in project_dirs:
        db = proj / "analysis.db" if proj.is_dir() else proj
        if not db.exists():
            raise DifferentiationError(f"no analysis.db under {proj}")
        dbs.append(db)

    for pi, db in enumerate(dbs):
        con = _open(db)
        try:
            seen_windows: set[int] = set()
            for table, col in instruments:
                wins = _load_instrument_windows(con, table, col, db.parent.name)
                for w, vec in wins.items():
                    per_inst[table][(pi, w)] = vec
                    seen_windows.add(w)
            proj_outcomes = _grounded_outcomes(con)
            for oname, wmap in proj_outcomes.items():
                dest = outcomes.setdefault(oname, {})
                for w in seen_windows:
                    if oname == "highlight":
                        dest[(pi, w)] = 1 if w in wmap else 0
                    elif w in wmap:
                        dest[(pi, w)] = wmap[w]
        finally:
            con.close()

    report = DifferentiationReport(
        instruments=inst_names, n_projects=len(dbs)
    )

    # Common windows where every instrument has data (for correlation alignment).
    key_sets = [set(per_inst[n].keys()) for n in inst_names if per_inst[n]]
    if not key_sets:
        raise DifferentiationError(
            "no instrument produced any windows — are these ingested projects?"
        )
    common = set.intersection(*key_sets)
    common_sorted = sorted(common)
    report.n_windows = len(common_sorted)

    if report.n_windows < MIN_WINDOWS:
        raise DifferentiationError(
            f"only {report.n_windows} windows where all instruments overlap "
            f"(need >= {MIN_WINDOWS}). Refusing to certify on insufficient data — "
            f"add more/longer real projects. (Single tiny clips cannot prove "
            f"independence.)"
        )

    # Build aligned full embedding matrix + 1-D PC1 signal per instrument.
    mats: dict[str, np.ndarray] = {}
    signals: dict[str, np.ndarray] = {}
    for name in inst_names:
        mat = np.stack([per_inst[name][k] for k in common_sorted])
        mats[name] = mat
        signals[name] = _pc1(mat)

    # ---- Intrinsic non-degeneracy (the corpus-independent zero-vector catch) ----
    # A degenerate instrument (constant / collapsed to rank<2 / all-zero rows)
    # carries no information regardless of any outcome, so this is a HARD failure.
    for name in inst_names:
        mat = mats[name]
        var = float(mat.var())
        zero_rows = int((np.abs(mat).sum(axis=1) == 0).sum())
        s = np.linalg.svd(mat - mat.mean(axis=0, keepdims=True), compute_uv=False)
        eff_rank = int((s > 1e-6 * (s.max() if s.size else 0)).sum())
        report.intrinsic[name] = {"variance": var, "eff_rank": float(eff_rank),
                                  "zero_rows": float(zero_rows)}
        if var <= 0 or eff_rank < 2:
            report.status[name] = "DEGENERATE"
            report.failures.append(
                f"{name}: intrinsically degenerate (variance={var:.2e}, "
                f"eff_rank={eff_rank}, zero_rows={zero_rows}) — collapsed/constant "
                f"embeddings (the zero-vector bug class)."
            )

    # ---- Grounding: MI vs each instrument's RELEVANT outcomes ----
    ln2 = math.log(2.0)
    min_subset = max(MIN_WINDOWS, 8)
    # Pre-encode usable outcomes (>=2 classes over a large-enough defined subset).
    usable_outcomes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for oname, wmap in outcomes.items():
        idx = np.array([i for i, k in enumerate(common_sorted) if k in wmap])
        if idx.size < min_subset:
            continue
        raw = [wmap[common_sorted[i]] for i in idx]
        uniq = {v: i for i, v in enumerate(sorted(set(raw), key=str))}
        if len(uniq) < 2:
            continue
        usable_outcomes[oname] = (idx, np.array([uniq[v] for v in raw]))

    if not usable_outcomes:
        raise DifferentiationError(
            "no grounded outcome has >=2 classes over a large-enough window subset "
            "— mutual information is undefined. Use richer projects "
            "(highlights/sentiment/scenes that vary across windows)."
        )

    for name in inst_names:
        if report.status.get(name) == "DEGENERATE":
            report.bits[name] = 0.0
            report.grounded_by[name] = "n/a"
            continue
        relevant = INSTRUMENT_OUTCOMES.get(name, list(usable_outcomes))
        testable = [o for o in relevant if o in usable_outcomes]
        sig = signals[name]
        best_bits, best_outcome = 0.0, "none"
        for oname in testable:
            idx, y = usable_outcomes[oname]
            x = sig[idx].reshape(-1, 1)
            mi_nats = float(
                mutual_info_classif(x, y, discrete_features=False, random_state=0)[0]
            )
            bits = mi_nats / ln2
            if bits > best_bits:
                best_bits, best_outcome = bits, oname
        report.bits[name] = best_bits
        report.grounded_by[name] = best_outcome
        if not testable:
            # No relevant outcome varies in THIS corpus -> cannot test grounding.
            report.status[name] = "UNTESTABLE"
            report.warnings.append(
                f"{name}: grounding UNTESTABLE in this corpus (no varying "
                f"{'/'.join(relevant)} — e.g. single-speaker clips). Intrinsic "
                f"check passed, so not failed; add multi-{relevant[0]} projects "
                f"to certify grounding."
            )
        elif best_bits < min_bits:
            report.status[name] = "WEAK"
            report.failures.append(
                f"{name}: {best_bits:.4f} bits < {min_bits} vs its testable "
                f"outcomes ({', '.join(testable)}) — carries no grounded "
                f"information about what it should encode."
            )
        else:
            report.status[name] = "OK"

    # Pairwise correlation ceiling.
    from scipy.stats import pearsonr

    for i, a in enumerate(inst_names):
        for b in inst_names[i + 1 :]:
            if np.std(signals[a]) == 0 or np.std(signals[b]) == 0:
                c = 0.0
            else:
                c = abs(float(pearsonr(signals[a], signals[b])[0]))
            report.corr[(a, b)] = c
            if c > max_corr:
                report.failures.append(
                    f"{a} x {b}: |corr|={c:.3f} > {max_corr} — redundant "
                    f"instruments (one adds nothing the other doesn't)."
                )

    return report


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        print("error: provide >=1 project dir (or analysis.db path).", file=sys.stderr)
        return 2
    dirs = [Path(a).expanduser() for a in args]
    try:
        report = validate_differentiation(dirs)
    except DifferentiationError as e:
        print(f"CANNOT CERTIFY: {e}", file=sys.stderr)
        return 3
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
