#!/usr/bin/env python3
"""
Create the full atomic GitHub-issue set for ClipCannon, each with a
Full State Verification (FSV) contract: source of truth, trigger->process->outcome,
a happy-path synthetic reality check, 5 failure edge cases, and an evidence command.

Scope: clipcannon repo (core pipeline, editing, rendering, audio, voice, phoenix
avatar, voiceagent/meeting bot, dashboard, billing, gpu) + NVIDIA-inspired upgrades.
Leapable.ai product build-out is intentionally OUT of scope.

Usage:
    python scripts/github_fsv_issues.py            # dry run (prints titles)
    python scripts/github_fsv_issues.py --create   # actually create labels + issues
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

REPO = "ChrisRoyse/clipcannon"

# ---------------------------------------------------------------------------
# Labels (idempotent)
# ---------------------------------------------------------------------------
LABELS = [
    ("area:pipeline", "1d76db", "Analysis pipeline / embedding instruments"),
    ("area:avatar", "5319e7", "Phoenix avatar engine + lip-sync"),
    ("area:voice", "0e8a16", "Voice cloning / TTS"),
    ("area:meeting", "006b75", "voiceagent / meeting bot"),
    ("area:editing", "fbca04", "EDL editing engine"),
    ("area:rendering", "c5def5", "FFmpeg/NVENC rendering"),
    ("area:audio", "bfdadc", "AI music / SFX / cleanup"),
    ("area:dashboard", "d4c5f9", "Web dashboard"),
    ("area:gpu", "b60205", "GPU / VRAM / precision"),
    ("area:testing", "0052cc", "Tests / FSV infrastructure"),
    ("area:packaging", "cfd3d7", "Install / deps / MCP wiring"),
    ("nvidia-upgrade", "76b900", "NVIDIA Blackwell / Cosmos / NVFP4 driven"),
    ("root-cause", "e11d21", "Root-cause fix, not a workaround"),
    ("blocker", "b60205", "Blocks a shipping capability"),
    ("fsv", "5319e7", "Has Full State Verification contract"),
    ("epic", "3e4b9e", "Tracking epic"),
    ("paper", "fef2c0", "Calculus-of-Association theory tie-in"),
]


def constraints() -> str:
    return textwrap.dedent(
        """\
        ## Constraints (non-negotiable)
        - **No workarounds or fallbacks.** If it cannot work, it must `raise` with robust,
          specific error logging: *what* failed, *where*, *why*, and *how to fix*.
        - **No mock data in tests.** Use real data or synthetic inputs with known expected
          outputs. Verify against the real source of truth, not return values alone.
        - **Never mask a broken state with a passing test.** A test must fail when the system
          is broken.
        - First-principles root cause only — do not patch the symptom."""
    )


def body(
    *,
    capability: str,
    root_cause: str,
    dod: list[str],
    sot: str,
    tpo: str,
    happy: str,
    edges: list[str],
    evidence: str,
    refs: str = "",
) -> str:
    dod_md = "\n".join(f"- [ ] {d}" for d in dod)
    edges_md = "\n".join(f"{i+1}. {e}" for i, e in enumerate(edges))
    refs_md = f"\n## References\n{refs}\n" if refs else ""
    return f"""## Capability — what the system needs from this
{capability}

## Root cause (first principles)
{root_cause}

## Definition of Done
{dod_md}

{constraints()}

## Full State Verification (FSV)
**Source of Truth:** {sot}

**Trigger → Process → Outcome:** {tpo}

**Happy path (reality check):** {happy}

**5 edge cases that would mean it is BROKEN:**
{edges_md}

**Evidence of success (must be attached to the closing comment):**
{evidence}
{refs_md}
🤖 Filed by Claude Code — atomic FSV issue."""


I = []  # (title, labels, body)


def add(title, labels, **kw):
    I.append((title, labels, body(**kw)))


# ===========================================================================
# A. INSTRUMENT PANEL  (paper: Calculus of Association, N=7)
# ===========================================================================

add(
    "Restore the dead emotion/valence instrument (zero-vector + valence=0.5 placeholder)",
    "area:pipeline,root-cause,blocker,fsv,paper",
    capability=(
        "The emotion instrument is 1 of the 7 frozen embedders that constitute the video "
        "panel. The paper's differentiation contract requires every instrument to add "
        "≥0.05 bits of information about a real outcome. Today this instrument is dead."
    ),
    root_cause=(
        "`src/clipcannon/pipeline/emotion_embed.py` (~line 257) emits a hardcoded "
        "`valence = 0.5` and a `np.zeros(EMBEDDING_DIM)` placeholder when valence cannot be "
        "derived. Result: vec_emotion rows are degenerate / constant, so the instrument "
        "contributes 0 bits and silently corrupts the constellation centroids and any "
        "emotion-conditioned editing/avatar behaviour."
    ),
    dod=[
        "Replace placeholder with a real valence+arousal estimator from the audio (no constant fill).",
        "If valence genuinely cannot be computed for a segment, RAISE with diagnostics — do not write a fake row.",
        "Embeddings written to vec_emotion have non-zero variance across segments.",
        "emotion_curve table is populated with varying valence/arousal across a real clip.",
        "Add real-data FSV test (no mocks).",
    ],
    sot="`analysis.db` tables `emotion_curve` and vector table `vec_emotion`; per-project DB under `~/.clipcannon/projects/<id>/`.",
    tpo="Trigger: `clipcannon_ingest` with `emotion_embed` stage enabled → Process: Wav2Vec2/valence model over vocal stem → Outcome: per-segment valence/arousal rows + emotion vectors persisted.",
    happy=(
        "Ingest a clip with one clearly happy passage and one clearly sad passage. "
        "Read emotion_curve: the happy span must show valence > 0.6 and the sad span valence < 0.4 "
        "(monotonic separation), and `SELECT COUNT(DISTINCT valence) FROM emotion_curve` > 1."
    ),
    edges=[
        "Silent / near-silent audio segment → must RAISE or mark stage failed with dBFS evidence, not write valence=0.5.",
        "Pure music, no speech → instrument must report it cannot derive vocal valence and fail loudly, not emit a neutral constant.",
        "All-zero vec_emotion variance after a real ingest → broken (the original bug).",
        "Two acoustically opposite clips produce identical valence → broken (constant output).",
        "Model checkpoint missing → must error with the exact missing path, never silently fall back to zeros.",
    ],
    evidence="`sqlite3 analysis.db \"SELECT t_start,valence,arousal FROM emotion_curve ORDER BY t_start LIMIT 20;\"` plus a numpy variance print of vec_emotion rows.",
    refs="Calculus-of-Association paper §differentiation contract (≥0.05 bits/instrument, no pair corr >0.6).",
)

add(
    "Implement the instrument differentiation-contract validator (≥0.05 bits MI, pairwise corr ≤0.6)",
    "area:pipeline,fsv,paper",
    capability=(
        "Operationalise the paper's contract so a degenerate instrument (like the valence bug) "
        "is caught automatically rather than discovered by accident. Turns the theory into a CI gate."
    ),
    root_cause=(
        "There is no automated check that each instrument adds independent, grounded information. "
        "Nothing measures mutual information vs a real outcome or pairwise redundancy."
    ),
    dod=[
        "Add a validator that, given a corpus of real ingested projects, computes per-instrument MI against a labelled outcome and pairwise correlation across the 7 (→8) instruments.",
        "Fail if any instrument < 0.05 bits or any pair > 0.6 correlation.",
        "Wire as an offline CLI + optional CI gate.",
    ],
    sot="Vector tables vec_frames/vec_semantic/vec_emotion/vec_speakers (+ prosody/sentiment/voice stores) across ≥3 real projects.",
    tpo="Trigger: run validator over N real projects → Process: MI + correlation matrix → Outcome: PASS/FAIL report with per-instrument bits and the correlation heatmap values.",
    happy="With the emotion bug fixed, run over 3 real clips: all 8 instruments report ≥0.05 bits and no pair >0.6; report exits 0.",
    edges=[
        "Re-introduce the zero-vector emotion bug → validator must FAIL on emotion instrument.",
        "Duplicate an instrument (feed same model twice) → must FAIL the >0.6 pair check.",
        "Single-project corpus (insufficient samples) → must refuse to certify and say why, not pass trivially.",
        "Outcome labels all identical → MI undefined → must error, not report 0 silently as pass.",
        "NaN/Inf in an embedding → must surface the offending project/row, not crash opaquely.",
    ],
    evidence="Printed bits-per-instrument table + correlation matrix + exit code, committed as a fixture report.",
    refs="Paper: Derived Data Abundance; differentiation contract.",
)

add(
    "Implement & persist cross-terms (N-choose-2 associations-between-associations)",
    "area:pipeline,paper",
    capability=(
        "Derived Data Abundance claims n inputs through N instruments yield up to "
        "n·(N + C(N,2) + 1) structured signals — the C(N,2) cross-terms are the associations "
        "between associations. Verify whether these are actually computed/stored, and implement if not."
    ),
    root_cause="Codebase stores per-instrument embeddings but it is unclear the cross-terms are materialised or used by discovery/editing tools.",
    dod=[
        "Audit whether cross-terms exist anywhere; document finding.",
        "If absent, compute and store the C(N,2) pairwise cross-association features for each input window.",
        "Expose them to at least one discovery tool (e.g. find_best_moments) and show measurable lift.",
    ],
    sot="A new/identified table (e.g. `cross_terms`) keyed by (window, instrument_i, instrument_j).",
    tpo="Trigger: ingest → Process: pairwise cross-association computation → Outcome: C(7,2)=21 (or C(8,2)=28) cross-term rows per window persisted.",
    happy="After ingest of a 60s clip windowed at 1s, cross_terms row count == windows × C(N,2); spot-check a known correlated pair (speaker×emotion) is non-trivial.",
    edges=[
        "Window with a missing instrument → cross-term must be flagged missing, not silently 0.",
        "N changes (8th instrument added) → count must update to C(8,2) automatically, not stay 21.",
        "Identical instruments → cross-term degenerate → must be caught by the contract validator.",
        "Empty project → 0 windows → 0 cross-terms, no crash.",
        "Cross-term count != windows×C(N,2) → broken bookkeeping.",
    ],
    evidence="`sqlite3 analysis.db \"SELECT COUNT(*) FROM cross_terms;\"` vs computed windows×C(N,2).",
    refs="Paper: Derived Data Abundance upper bound under approximate independence (capped by DPI).",
)

add(
    "Add 8th instrument: Cosmos-3 world/physics-reasoning embedder (grow panel N=7→8)",
    "area:pipeline,nvidia-upgrade,paper",
    capability=(
        "NVIDIA Cosmos 3 (Jun 2026) is an open omnimodal world model; the Cosmos reasoning VLM "
        "(2B/8B) runs on RTX/Jetson. Adding a physics/world-reasoning axis is exactly the "
        "'commission a new lens for a new axis' move the paper describes, and the new cross-terms "
        "bridge it to the existing 7 instruments."
    ),
    root_cause="No instrument captures physical-world / spatio-temporal plausibility of a scene; this axis is currently unmeasured.",
    dod=[
        "Integrate Cosmos-3 (2B or 8B) reasoning VLM as a frozen embedder producing a per-window vector + a physical-plausibility scalar.",
        "Persist to a new vec table; register with the differentiation validator (must clear ≥0.05 bits, ≤0.6 corr).",
        "No runtime download — model must be pre-cached or error.",
    ],
    sot="New vector table `vec_world` (Cosmos embedding) + per-window plausibility score.",
    tpo="Trigger: ingest with cosmos stage → Process: Cosmos VLM over sampled frames → Outcome: vec_world rows + plausibility persisted.",
    happy="Ingest a normal clip and a temporally-scrambled version of it; plausibility for scrambled < normal; vec_world variance non-zero.",
    edges=[
        "VRAM insufficient for chosen Cosmos size → must error with required-vs-available GB, not OOM-crash.",
        "Cosmos checkpoint not pre-cached → error with exact path (no auto-download).",
        "vec_world correlates >0.6 with SigLIP visual → flag redundancy via validator (instrument adds nothing).",
        "Frame sample rate 0 → must refuse, not divide-by-zero.",
        "Scrambled clip scores ≥ normal → instrument not measuring what we claim → broken.",
    ],
    evidence="vec_world variance print + plausibility(normal) vs plausibility(scrambled) comparison logged.",
    refs="NVIDIA Cosmos 3 (research.nvidia.com/labs/cosmos-lab/cosmos3); Cosmos VLM 2B/8B on Jetson.",
)

add(
    "Unit + FSV tests for the 7 embedding stages (real data, no mocks)",
    "area:pipeline,area:testing,fsv",
    capability="Each of the 7 instruments must be independently proven to write correct, non-degenerate vectors to its store on real input.",
    root_cause="The 7 embedding stages have integration coverage but no per-stage real-data FSV asserting the vectors actually land with correct shape/variance.",
    dod=[
        "Per-stage test for visual(SigLIP 1152), semantic(Nomic 768), emotion(Wav2Vec2 1024), speaker(WavLM 512), prosody(12), sentiment(MiniLM 384), voice(ECAPA 192).",
        "Each asserts: row count == expected windows, dim == expected, variance > 0, no NaN.",
        "Use a committed real fixture clip with known structure.",
    ],
    sot="vec_frames / vec_semantic / vec_emotion / vec_speakers + prosody/sentiment/voice tables.",
    tpo="Trigger: run stage on fixture → Process: model inference → Outcome: vectors persisted with correct dim/shape.",
    happy="Fixture clip (2 speakers, 1 happy + 1 sad span) → speaker table shows 2 clusters; visual dim==1152; emotion variance>0.",
    edges=[
        "Wrong embedding dim written → fail.",
        "Constant vector across windows → fail (variance≈0).",
        "NaN/Inf in any vector → fail with row id.",
        "Row count mismatch vs windows → fail.",
        "Model loads on CPU when CUDA expected → fail (precision/perf regression).",
    ],
    evidence="Per-stage SELECT of dim + variance + count from the relevant vec table.",
)

add(
    "Unit + FSV tests for the 17 non-embedding pipeline stages",
    "area:pipeline,area:testing,fsv",
    capability="Probe, VFR-normalize, audio-extract, frame-extract, transcribe, ocr, quality, shot_type, scene_analysis, storyboard, source_separation, reactions, acoustic, chronemic, profanity, highlights, finalize must each be independently verified against their persisted outputs.",
    root_cause="These stages currently lack dedicated real-data tests asserting their table writes.",
    dod=[
        "One FSV test per stage asserting its specific table/file output on a real fixture.",
        "transcribe: word + segment rows with monotonic timestamps; source_separation: 4 stem files exist & non-silent.",
        "highlights: scores present and ordered; scene_analysis: ≥1 boundary on a multi-scene clip.",
    ],
    sot="Respective tables (transcript_segments/words, scenes, highlights, acoustic, ...) + stem WAV files on disk.",
    tpo="Trigger: run stage → Process: stage logic → Outcome: rows/files persisted.",
    happy="Fixture with 2 visual scenes and clear speech → scenes≥2, transcript_words>0 with increasing t, 4 demucs stems written.",
    edges=[
        "VFR input → vfr_normalize must produce CFR; assert constant frame delta.",
        "Silent audio → transcribe fails-fast in ~0.01s with dBFS diagnostic (locked decision).",
        "Demucs failure → full stderr surfaced in stream_status.error_message, not swallowed.",
        "Single-scene clip → scene_analysis must not hallucinate boundaries.",
        "Corrupt frame → quality stage must report it, not crash the DAG.",
    ],
    evidence="Per-stage SELECT/`ls` proof + a stream_status dump showing stage states.",
)

add(
    "Lock the silent-audio fail-fast decision with a regression FSV test",
    "area:pipeline,area:testing,fsv",
    capability="A previously-fixed root cause (silent audio passing) must never regress; the decision is locked.",
    root_cause="`transcribe.py` now rejects silent audio in ~0.01s before loading WhisperX, but there is no permanent guard against regression.",
    dod=[
        "Real-data test with a synthetic -91 dBFS AAC stream that asserts fail-fast with the full diagnostic.",
        "Assert it fails BEFORE the WhisperX load cost.",
    ],
    sot="`stream_status` row for the transcribe stage + the raised error's diagnostic fields.",
    tpo="Trigger: ingest silent video → Process: _reject_if_silent() → Outcome: stage marked failed with peak dBFS/RMS/audible%/remediation.",
    happy="Synthetic silent clip → transcribe fails in <0.5s; error message contains peak dBFS and remediation text.",
    edges=[
        "Borderline -50 dBFS audible clip → must PASS (not over-reject).",
        "Loud clip → passes normally.",
        "Diagnostic missing any of dBFS/RMS/audible%/remediation → fail.",
        "Failure takes >5s (WhisperX loaded) → regression, fail.",
        "Silent clip silently producing empty transcript → broken.",
    ],
    evidence="Timed log of the failure + the diagnostic payload.",
    refs="memory/decisions/agent-00-coordinator--fail-fast-on-silent-audio.md",
)

# ===========================================================================
# B. AVATAR / LIP-SYNC
# ===========================================================================

add(
    "[EPIC] Real-time + offline avatar lip-sync: remove all stubs, ship a verified path",
    "epic,area:avatar,blocker",
    capability="A clone must move its mouth in sync with generated speech, in both offline video generation and live meetings. Today the realtime path is a stub that raises.",
    root_cause="MuseTalk realtime inference is unwired; EchoMimic is blocked by mmcv/sm_120; only physics-based fallback + idle driver loop exist for live.",
    dod=[
        "Pick ONE shipping realtime path and ONE offline path; remove every stub that raises NotImplemented/placeholder.",
        "Child issues: Audio2Face-3D integration, MuseTalk decision, EchoMimic mmcv root cause, hardcoded path, identity guard.",
    ],
    sot="Rendered MP4 output frames + v4l2loopback device stream; lip-open metric vs audio envelope.",
    tpo="Trigger: generate_video / meeting speech → Process: audio→lip-sync model→composited frames → Outcome: synced video.",
    happy="Generate a 5s talking-head saying a known phrase; mouth-open correlates with audio RMS envelope (corr>0.5); no frame where stub error is hit.",
    edges=[
        "Silent audio → mouth closed, no jitter.",
        "Audio longer than driver video → handled without freeze.",
        "Non-English audio → lips still track (physics path).",
        "Identity drift across frames → guard must reject (see ArcFace issue).",
        "Any code path still raising the stub error → epic not done.",
    ],
    evidence="Side-by-side audio-envelope vs mouth-open plot + output file inspection.",
)

add(
    "Integrate Audio2Face-3D diffusion lip-sync (Blackwell-native) via existing audio2face_adapter",
    "area:avatar,nvidia-upgrade,blocker,fsv",
    capability="NVIDIA's diffusion-based Audio2Face-3D improves lip/tongue movement and is Blackwell-native (no mmcv). The repo already has `src/phoenix/adapters/audio2face_adapter.py` (FLAME↔ARKit 52). This is the cleanest unblock for lip-sync.",
    root_cause="Lip-sync currently depends on either the stubbed MuseTalk or formant-physics only; no production diffusion lip model is wired through the existing adapter.",
    dod=[
        "Wire Audio2Face-3D → ARKit/FLAME blendshapes → Phoenix compositor.",
        "Pre-cached model only (no runtime download); error if missing.",
        "Replace the raising MuseTalk stub call sites with this path.",
    ],
    sot="ARKit blendshape stream + composited frames; jaw_open/lip channels per frame.",
    tpo="Trigger: audio chunk → Process: Audio2Face-3D → blendshapes → CuPy composite → Outcome: synced frame to output/loopback.",
    happy="Known phrase audio → blendshape jaw_open time-series correlates with audio envelope; tongue/lip channels non-constant on plosives/fricatives.",
    edges=[
        "Missing checkpoint → exact-path error, no download.",
        "Audio sample rate mismatch → resample explicitly or error, never silent garbage.",
        "Blendshape values out of [0,1] → clamp+log or error, not NaN frames.",
        "Frame rate drift vs audio → A/V desync must be detected.",
        "All-zero blendshapes on speech → broken, fail.",
    ],
    evidence="Per-frame blendshape CSV + envelope correlation + sample output frames.",
    refs="NVIDIA Audio2Face-3D diffusion model (CES 2026 RTX AI). Existing adapter: src/phoenix/adapters/audio2face_adapter.py.",
)

add(
    "Decide & resolve MuseTalk 1.5 realtime: wire it or delete the stub (no raising placeholder)",
    "area:meeting,area:avatar,root-cause,blocker,fsv",
    capability="The meeting avatar's realtime lip-sync must either work or not exist — a method that lazy-loads then raises MeetingLipSyncError is the worst state.",
    root_cause="`src/voiceagent/meeting/avatar_rt.py:~92` raises 'MuseTalk inference not yet integrated'; `:~61` placeholder init.",
    dod=[
        "Decision recorded: keep MuseTalk (wire process_audio_chunk) OR drop in favour of Audio2Face-3D.",
        "If kept: real 256x256 lip frames generated at target FPS. If dropped: remove module + references cleanly.",
        "No remaining code path raises a 'not integrated' error.",
    ],
    sot="RealtimeLipSync output frames + meeting webcam_writer v4l2 device.",
    tpo="Trigger: meeting TTS audio active → Process: lip-sync model → Outcome: synced frames on virtual cam.",
    happy="Live meeting with TTS speech → mouth tracks audio at ≥24 FPS; grep of repo shows zero 'not yet integrated' raises.",
    edges=[
        "TTS chunk arrives faster than render → backpressure handled, no crash.",
        "Model not loaded → explicit error before meeting join, not mid-call.",
        "FPS < target → logged + measured, not silently degraded.",
        "Speaking but idle-loop shown (no lip motion) → broken.",
        "Stub error reachable in any branch → not done.",
    ],
    evidence="FPS log + frame samples during speech; `grep -rn 'not yet integrated' src/` returns nothing.",
)

add(
    "Root-cause EchoMimicV3 mmcv ↔ RTX 5090 sm_120 incompatibility",
    "area:avatar,area:gpu,root-cause,blocker,nvidia-upgrade",
    capability="Either make EchoMimic build for Blackwell or formally migrate off it — but stop carrying a blocked training path.",
    root_cause="mmcv (EchoMimic dep) is compiled for sm_90 and has no forward-compat for Blackwell sm_120; training pipeline is inert. NVIDIA's Blackwell-native push (FP4, Audio2Face, LTX-2) is the strategic exit.",
    dod=[
        "Investigate building mmcv from source for sm_120 vs dropping mmcv entirely.",
        "Recommendation recorded with evidence (build log or migration plan).",
        "If migrating: remove mmcv from deps; clone-video path uses a Blackwell-native model.",
    ],
    sot="Working `import mmcv` on sm_120 OR a clone-video render produced without mmcv.",
    tpo="Trigger: clone_video → Process: chosen avatar model → Outcome: identity-locked MP4.",
    happy="Clone-video of the Santa profile renders end-to-end on the 5090 with no mmcv import error; output passes identity guard.",
    edges=[
        "mmcv build succeeds but crashes at inference on sm_120 → still broken.",
        "Migration path produces lower identity score → must be measured, not hidden.",
        "uv.lock still pins mmcv after migration → not done.",
        "Falls back to CPU → fail (perf), surface it.",
        "Render silently uses a different person → identity guard must catch.",
    ],
    evidence="Build/import log on sm_120 OR a clean clone-video render log + identity score.",
    refs="memory: project_musetalk_setup (mmcv blocked), feedback_echomimic_vram_bf16.",
)

add(
    "Remove hardcoded /home/cabdru/echomimic_v3 path → config/env (portability)",
    "area:avatar,root-cause,fsv",
    capability="Clone video generation must run on any install, not only this machine.",
    root_cause="`src/clipcannon/tools/constellation.py:~140` hardcodes `echomimic_dir = \"/home/cabdru/echomimic_v3\"`.",
    dod=[
        "Path comes from config/env with validation + clear error if unset/missing.",
        "Grep shows no hardcoded /home/cabdru paths in src/.",
    ],
    sot="Config value + the resolved path used at runtime; constellation.py source.",
    tpo="Trigger: clone_video on a fresh checkout → Process: resolve model dir from config → Outcome: runs or errors with the config key to set.",
    happy="Unset config → error names the exact env/config key; set it → render proceeds.",
    edges=[
        "Config points to nonexistent dir → explicit error with path.",
        "Relative path → resolved to absolute deterministically.",
        "Trailing slash / symlink → handled.",
        "Any remaining `/home/cabdru` literal in src → fail.",
        "Works on this machine only → not done.",
    ],
    evidence="`grep -rn '/home/cabdru' src/` returns nothing + fresh-env run log.",
)

add(
    "Enforce identity-lock via ArcFace face-crop verification in the constellation guard",
    "area:avatar,fsv",
    capability="Generated clone frames must be the right person. Per prior finding, SigLIP global embeddings fail identity (an interviewer scored 0.985 vs Santa); ArcFace on face crops is required.",
    root_cause="constellation_guard verifies frames but identity verification must use ArcFace on detected face crops, not a global SigLIP embedding.",
    dod=[
        "Guard runs ArcFace on the detected face crop each generated frame (or sampled).",
        "Reject + log frames below an identity threshold calibrated on real reference.",
        "Threshold chosen from real positive/negative pairs, documented.",
    ],
    sot="Per-frame identity score log + rejected-frame records.",
    tpo="Trigger: clone frame generated → Process: face detect → ArcFace crop embedding → cosine vs reference → Outcome: accept/reject decision logged.",
    happy="Generate Santa: identity score > threshold for true frames; feed an interviewer frame → score below threshold → rejected.",
    edges=[
        "No face detected → frame rejected/flagged, not auto-accepted.",
        "Two faces in frame → correct (largest/centered) face scored.",
        "SigLIP-only path still used anywhere → fail (must be ArcFace).",
        "Interviewer/wrong person scores above threshold → broken.",
        "Threshold hardcoded without calibration evidence → not done.",
    ],
    evidence="ROC/threshold note + a positive vs negative score comparison log.",
    refs="memory: feedback_identity_verification_arcface.",
)

add(
    "LatentSync offline lip-sync FSV (no CodeFormer; locked best params)",
    "area:avatar,fsv",
    capability="The offline LatentSync path must produce the known-best quality and never re-introduce the blur-amplifying CodeFormer step.",
    root_cause="Best params (20 steps, 1.5 guidance, 25fps, audio remux, NO CodeFormer) are known from prior work but not guarded by a test.",
    dod=[
        "FSV test asserts params and that no CodeFormer pass runs on output.",
        "Audio is remuxed (A/V aligned).",
    ],
    sot="Output MP4 (codec/fps/streams) + the pipeline config actually used.",
    tpo="Trigger: lip_sync tool → Process: LatentSync 1.6 → Outcome: synced MP4 at 25fps with original audio remuxed.",
    happy="Run lip_sync on fixture → output is 25fps, audio stream present & aligned, mouth tracks envelope, no CodeFormer in the call graph.",
    edges=[
        "CodeFormer enabled anywhere → fail.",
        "Output fps != 25 → fail.",
        "Audio missing/desynced in output → fail.",
        "Guidance/steps drifted from locked values → fail.",
        "Blur worse than reference baseline → measured + fail.",
    ],
    evidence="`ffprobe` of output + call-graph grep for codeformer + envelope correlation.",
    refs="memory: feedback_lipsync_quality, feedback_lipsync_quality_v2.",
)

# ===========================================================================
# C. VOICE
# ===========================================================================

add(
    "Voice clone SECS>0.95 hard gate FSV against the REAL reference voice (full ICL)",
    "area:voice,fsv",
    capability="Cloned voice must match the real person at SECS>0.95 with full prosody; this is a hard quality floor, no trade-offs for speed.",
    root_cause="Quality gate exists but must be proven to score against a real reference recording (full ICL + ref_text), not a generic/denoised proxy.",
    dod=[
        "FSV test clones a real reference and asserts SECS>0.95 vs that reference.",
        "Failing candidates RAISE MeetingVoiceError — no degraded generic voice emitted.",
        "Full ICL mode with ref_text confirmed in the call path.",
    ],
    sot="`~/.clipcannon/voice_profiles.db` voice_profiles row + the measured SECS in the generation log.",
    tpo="Trigger: speak/speak_optimized → Process: Qwen3-TTS clone + speaker-encoder SECS → Outcome: audio + SECS persisted; <0.95 ⇒ raise.",
    happy="Clone the 'boris' reference, synthesize a sentence → SECS>0.95 vs real boris recording; DNSMOS reasonable.",
    edges=[
        "All N candidates <0.95 → raise, no audio returned (no generic fallback).",
        "Reference too short/noisy → error with reason, not a low-quality clone.",
        "ICL ref_text missing → must error (accent/cadence would degrade).",
        "SECS computed vs wrong/denoised reference → invalid, fail.",
        "Returned audio is generic TTS voice → broken.",
    ],
    evidence="Generation log with per-candidate SECS + the stored profile row.",
    refs="memory: feedback_voice_quality_floor, feedback_voice_cloning.",
)

add(
    "Best-of-N optimizer FSV: winner selection persists correct profile + SECS",
    "area:voice,area:testing,fsv",
    capability="speak_optimized must actually pick and persist the highest-SECS candidate.",
    root_cause="Best-of-N ranking exists but isn't FSV-verified end to end (selection → persistence).",
    dod=["Test asserts the persisted output == argmax-SECS candidate across temperatures."],
    sot="voice_profiles.db / generation output file + per-candidate SECS log.",
    tpo="Trigger: speak_optimized(N=12) → Process: multi-temp generation + SECS rank → Outcome: best candidate saved.",
    happy="Run N=12 → saved file's SECS equals the max of the 12 logged SECS values.",
    edges=[
        "Tie in SECS → deterministic tie-break, documented.",
        "N=1 → still works.",
        "A candidate crashes → others still ranked, error logged.",
        "Saved file != argmax → broken.",
        "SECS recomputed on save differs from selection-time → consistency bug.",
    ],
    evidence="Per-candidate SECS table + final saved-file SECS.",
)

add(
    "Capture prosody features from vocal stems at ingest (for cloning references)",
    "area:voice,area:pipeline,fsv",
    capability="Ingest should extract prosody (F0, energy, rate, pitch contour) from separated vocal stems so voice cloning has grounded prosodic references.",
    root_cause="Prosody capture from vocal stems at ingest is desired but may be incomplete; cloning references lack stored prosody.",
    dod=[
        "prosody_segments populated from the demucs vocal stem (not the full mix).",
        "Stored features usable by the prosody matcher.",
    ],
    sot="`prosody_segments` table + the vocal stem WAV.",
    tpo="Trigger: ingest with prosody stage → Process: F0/energy/rate over vocal stem → Outcome: prosody rows persisted.",
    happy="Ingest expressive speech → prosody_segments F0 varies with intonation; computed from vocal stem, not mix.",
    edges=[
        "Run on full mix instead of vocal stem → fail (music contaminates F0).",
        "Unvoiced segment → F0 marked unvoiced, not a bogus number.",
        "No vocal stem (instrumental) → stage reports no prosody, not zeros.",
        "Constant F0 on varied speech → broken extractor.",
        "Rows not linkable to time spans → unusable, fail.",
    ],
    evidence="`SELECT t_start,f0,energy,rate FROM prosody_segments LIMIT 20;` + stem provenance.",
    refs="memory: project_prosody_capture.",
)

add(
    "FastTTS realtime latency FSV harness (~100-150ms TTFB, chunk streaming)",
    "area:voice,area:meeting,area:testing,fsv",
    capability="The realtime 0.6B TTS must meet its latency budget for live meetings.",
    root_cause="TTFB/chunk latency claimed but not measured by a repeatable harness.",
    dod=["Harness measures TTFB and per-chunk latency over real synthesis; asserts budget."],
    sot="Timestamped audio-chunk emission log.",
    tpo="Trigger: fast_tts synthesize → Process: CUDA-graph 0.6B TTS streaming → Outcome: first chunk emitted within budget.",
    happy="Synthesize a sentence → TTFB ≤150ms (warm), subsequent chunks ≤~170ms each.",
    edges=[
        "Cold start (model not resident) → measured separately, not hidden in warm number.",
        "Voice switch mid-stream → no large stall.",
        "Long input → streaming stays bounded, no full-buffer wait.",
        "GPU contention → degradation surfaced, not silently >1s.",
        "Reported TTFB excludes model load it actually incurred → invalid measurement.",
    ],
    evidence="Latency histogram / per-chunk timestamp log.",
    refs="memory: project_voiceagent_latency_optimization, project_voiceagent_tts.",
)

# ===========================================================================
# D. MEETING / VOICEAGENT
# ===========================================================================

add(
    "[EPIC] Meeting bot completeness (Mode 1 verified, Mode 2 decided)",
    "epic,area:meeting",
    capability="The clone must reliably participate in a live meeting end to end.",
    root_cause="Mode 1 (virtual device) is implemented; Mode 2 (bot join) is a placeholder; no E2E FSV exists.",
    dod=["Child issues: E2E pipeline FSV, Mode-2 decision, reasoning controller FSV, wake listener, RAG store, AEC."],
    sot="Meeting transcript store + virtual mic/cam streams + reasoning decision log.",
    tpo="Trigger: join meeting → Process: capture→ASR→reason→respond→speak→lip-sync → Outcome: clone speaks when addressed.",
    happy="Synthetic meeting audio addressing the clone → clone responds within ~2s with cloned voice + lip motion.",
    edges=["Not addressed → stays silent.","Overlapping speakers → correct turn-taking.","Network/device drop → clean error.","Wrong-person voice → fail.","Any stub raise → not done."],
    evidence="End-to-end run log + transcript store rows + A/V capture.",
)

add(
    "Mode-2 bot-join: implement browser WebRTC join OR remove placeholder (decision)",
    "area:meeting,root-cause,fsv",
    capability="Either the clone can join a meeting as a bot participant, or that mode is cleanly removed — not left as a placeholder method.",
    root_cause="`manager.py:start_clone_bot_join` and `browser_bot.py`/`bot_server.py` are placeholders; cli advertises a mode that doesn't work.",
    dod=[
        "Decision recorded. If implementing: WebRTC join to Meet/Zoom with audio+video tracks. If removing: delete method + CLI flag + skeleton files.",
        "No placeholder-only public method remains.",
    ],
    sot="Either a real joined-meeting session (participant visible) OR repo grep showing the mode is gone.",
    tpo="Trigger: start_clone_bot_join(url) → Process: browser/WebRTC → Outcome: bot present with A/V, or command no longer exists.",
    happy="If implemented: bot joins a test meeting, appears as participant, audio+video tracks publish.",
    edges=["Invalid URL → explicit error.","Join denied/waiting room → handled state, not hang.","No camera track → fail visibly.","Placeholder string remains → not done.","CLI lists a mode that errors → not done."],
    evidence="Joined-session screenshot/log OR `grep -rn placeholder src/voiceagent` clean.",
)

add(
    "E2E meeting pipeline FSV with synthetic audio (audio→ASR→reason→respond→speak)",
    "area:meeting,area:testing,fsv",
    capability="Prove the whole live loop wires together on real/synthetic audio without mocks.",
    root_cause="No end-to-end test exercises capture→transcribe→address-detect→respond→speak.",
    dod=["Synthetic WAV that addresses the clone drives the full pipeline; each stage's output verified."],
    sot="transcript store rows + responder output + voice_output audio + reasoning decision log.",
    tpo="Trigger: feed synthetic 'Hey [clone], what's 2+2?' → Process: full pipeline → Outcome: clone speaks an answer containing '4'.",
    happy="Synthetic question with a known answer → transcript captured, address detected true, response text contains the expected answer, audio emitted with SECS>0.95.",
    edges=[
        "Audio NOT addressing clone → address_detector false, no response.",
        "Empty/garbled audio → handled, logged, no crash.",
        "Responder LLM down → explicit error, not silent silence.",
        "Response emitted but never spoken (TTS gate fail) → surfaced.",
        "Pipeline 'passes' with no actual audio output → broken.",
    ],
    evidence="Stage-by-stage log + transcript rows + output WAV + recognized answer text.",
)

add(
    "Reasoning controller Tier1/2/3 FSV on synthetic situational scenarios",
    "area:meeting,area:testing,fsv",
    capability="The 3-tier controller must classify ActionIntent correctly so the avatar behaves appropriately.",
    root_cause="Tier1/2 decision logic has limited tests; ActionIntent accuracy unmeasured.",
    dod=["Synthetic SituationalAwareness inputs → assert expected ActionIntent for each canonical scenario."],
    sot="Reasoning decision log (intent + rationale) per tick.",
    tpo="Trigger: scripted situational frames → Process: Tier1/2 classification → Outcome: ActionIntent emitted.",
    happy="Direct question to clone → THINK→RESPOND; humor detected → LISTEN_AMUSED; nobody talking → IDLE.",
    edges=[
        "Sad content → LISTEN_EMPATHETIC not REACT_LAUGH.",
        "Long silence while addressed → eventually RESPOND, not infinite THINK.",
        "Rapid speaker changes → no thrashing between intents every tick.",
        "Ambiguous addressing → conservative (LISTEN), not false INTERJECT.",
        "Always RESPOND regardless of input → broken classifier.",
    ],
    evidence="Scenario→expected vs actual intent table.",
    refs="memory: project_reasoning_controller.",
)

add(
    "Wake listener 'Hey Jarvis' always-on auto-start FSV",
    "area:meeting,fsv",
    capability="The wake listener must auto-start with ClipCannon and reliably trigger on 'Hey Jarvis'.",
    root_cause="Wake listener must auto-start (per requirement) and detect via VAD→tiny-whisper→embedding similarity; needs verification it actually launches and triggers.",
    dod=["Auto-start hook verified; wake phrase triggers agent; 'go to sleep' returns to listening."],
    sot="Listener process state + wake-event log + spawned agent process.",
    tpo="Trigger: speak 'Hey Jarvis' → Process: VAD→ASR→similarity(≥0.72) → Outcome: agent subprocess launched.",
    happy="Play 'Hey Jarvis' WAV → wake event logged, agent starts; play 'go to sleep' → agent exits, listener resumes.",
    edges=[
        "Random speech (no wake phrase) → no trigger.",
        "Near-miss 'hey travis' below threshold → no false wake.",
        "Listener not auto-started on launch → fail (requirement).",
        "Agent fails to spawn → error logged, listener survives.",
        "Wake fires but agent never starts → broken.",
    ],
    evidence="Wake-event log + before/after process list.",
    refs="memory: feedback_wake_listener_always_on.",
)

add(
    "Leapable RAG meeting-store FSV (inline base64 ingest, 8+ word search retrieval)",
    "area:meeting,fsv",
    capability="Meeting transcripts must persist to Leapable and be retrievable so the clone has cross-session memory.",
    root_cause="Store migrated to Leapable (Windows host, inline base64 ingest, no session-id on init, 8+ word queries); retrieval correctness must be verified.",
    dod=["Ingest a transcript via inline base64; retrieve it with an 8+ word query and assert the known passage returns."],
    sot="Leapable store contents (via its query API) + ingest acknowledgement.",
    tpo="Trigger: post-meeting store → Process: base64 ingest to Leapable → Outcome: passage retrievable by semantic search.",
    happy="Ingest a transcript containing a unique known sentence; query with an 8+ word paraphrase → that passage is in top results.",
    edges=[
        "Short (<8 word) query → handled per Leapable constraint, not silently empty.",
        "Session-id passed on init → must NOT be sent (per constraint).",
        "Leapable host unreachable → explicit error, transcript not lost.",
        "Ingest returns ok but query finds nothing → broken (verify the read).",
        "Base64 corruption → detected, not stored garbled.",
    ],
    evidence="Ingest receipt + query results showing the known passage.",
    refs="memory: feedback_leapable_meeting_store, project_clone_meeting_agent.",
)

add(
    "Acoustic echo cancellation (AEC) FSV in WSL2/PulseAudio",
    "area:meeting,fsv",
    capability="The clone must not hear/transcribe its own TTS output (echo) during meetings.",
    root_cause="AEC filter + echo reference exist but WSL2/PulseAudio routing makes self-echo a real risk; unverified.",
    dod=["With TTS playing, the captured/transcribed stream does not contain the clone's own speech."],
    sot="Transcriber output during self-speech + AEC reference alignment log.",
    tpo="Trigger: clone speaks while capturing → Process: AEC removes reference → Outcome: ASR ignores own voice.",
    happy="Clone says a unique phrase; transcriber of the capture loop does NOT log that phrase as incoming speech.",
    edges=[
        "Reference delay misaligned → echo leaks → must be detected.",
        "Double-talk (clone + human simultaneously) → human still captured.",
        "AEC off → echo present → test must catch it.",
        "Clone transcribes itself and 'responds' to itself → broken loop.",
        "Comfort noise mistaken for speech → false ASR.",
    ],
    evidence="Capture-side transcript during self-speech showing no self-phrase.",
)

# ===========================================================================
# E. GPU / BLACKWELL STACK
# ===========================================================================

add(
    "[EPIC] Blackwell-native model stack: NVFP4/FP8, drop mmcv, TensorRT",
    "epic,area:gpu,nvidia-upgrade",
    capability="Exploit RTX 5090 (sm_120): NVFP4/FP8 quantization (2x perf, half VRAM) and TensorRT to fit the full concurrent pipeline without OOM.",
    root_cause="Models run in BF16/FP16; VRAM pressure forces serialization and blocks concurrency that the 5090 should allow.",
    dod=["Child issues: NVFP4 quantization, concurrent-load stress FSV, precision auto-detect FSV, TensorRT integration."],
    sot="vram_stats() readings + per-model precision report + throughput.",
    tpo="Trigger: load model stack → Process: FP4/FP8 quantized load → Outcome: lower VRAM, higher tok/s, same quality gates.",
    happy="Quantized stack fits concurrently in <28GB and passes all quality gates (SECS, identity, etc.).",
    edges=["Quality regresses past gate → fail.","FP4 unsupported op → explicit error.","VRAM still OOMs → not done.","Numerics NaN → caught.","CPU fallback → fail."],
    evidence="VRAM + throughput + quality-gate comparison BF16 vs FP4.",
    refs="NVIDIA RTX Blackwell FP4/NVFP4 (LTX-2/FLUX.2 articles, Jun 2026).",
)

add(
    "NVFP4 quantization for Blackwell models (2x perf / half VRAM) with quality-gate FSV",
    "area:gpu,nvidia-upgrade,fsv",
    capability="Quantize the heaviest models to NVFP4 on sm_120 to relieve VRAM and speed inference, WITHOUT crossing any quality floor.",
    root_cause="No FP4 path; the 5090's 5th-gen Tensor Core FP4 support is unused.",
    dod=[
        "Quantize selected models (e.g. TTS, video/lip-sync) to NVFP4.",
        "Assert VRAM drop ~50% and throughput up ~2x.",
        "Assert SECS/identity/quality gates still pass on real data.",
    ],
    sot="vram_stats() before/after + quality metrics (SECS, identity score) on real fixtures.",
    tpo="Trigger: load model FP4 → Process: quantized inference → Outcome: lower VRAM + faster + gates intact.",
    happy="TTS at NVFP4: VRAM ~half of BF16, TTFB lower, SECS still >0.95 on the real reference.",
    edges=[
        "SECS drops below 0.95 under FP4 → must FAIL, not ship degraded.",
        "FP4 op unsupported on a layer → explicit error, no silent FP32 promotion that erases the win.",
        "VRAM not actually lower → quantization not applied → fail.",
        "Throughput unchanged → not done.",
        "NaN/Inf activations → caught and reported.",
    ],
    evidence="Before/after vram_stats + SECS comparison table.",
)

add(
    "GPU concurrent-load stress FSV on RTX 5090 (no OOM, no WSL2 crash, serialized loads)",
    "area:gpu,area:testing,fsv",
    capability="The 32GB 5090 must run the intended concurrent set (ASR+LLM+TTS, or analysis stack) without OOM or WSL2 CUDA crashes.",
    root_cause="Concurrent CUDA contexts can crash WSL2; load serialization + aggressive cleanup is required but unverified under stress.",
    dod=["Stress test loads the concurrent budget repeatedly; asserts no OOM, no crash, VRAM under budget."],
    sot="vram_stats() peak + process survival + dmesg/WSL stability.",
    tpo="Trigger: load concurrent models in a loop → Process: serialized loads + cleanup → Outcome: stable under budget.",
    happy="Load ASR+LLM+TTS 20x with cleanup → peak VRAM < budget, no crash, all loads succeed.",
    edges=[
        "Parallel (non-serialized) loads → reproduce the WSL crash, prove serialization fixes it.",
        "VRAM exceeds budget → check_vram raises before OOM.",
        "Leak across iterations (VRAM grows) → caught.",
        "ipc_collect not freeing → detected.",
        "Crash logged as success → broken.",
    ],
    evidence="Per-iteration VRAM log + survival confirmation.",
    refs="memory: feedback_wsl_gpu_stability.",
)

add(
    "Precision auto-detect FSV (BF16 on sm_120 vs FP16 elsewhere)",
    "area:gpu,area:testing,fsv",
    capability="The GPU manager must pick the correct precision per compute capability.",
    root_cause="Auto-detect logic exists (Blackwell→BF16/TF32) but isn't asserted.",
    dod=["Test asserts selected precision matches detected compute capability."],
    sot="GPU manager health report (precision field).",
    tpo="Trigger: init on 5090 → Process: capability detect → Outcome: BF16+TF32 selected.",
    happy="On sm_120 → report says BF16; TF32 matmul enabled.",
    edges=["Spoof Ampere → FP16/INT8 path.","Unknown GPU → safe default + warning, not crash.","CPU-only → FP32 reported.","Wrong precision selected → fail.","TF32 flags not set on Blackwell → fail."],
    evidence="GPU manager health JSON.",
)

add(
    "Evaluate + integrate TensorRT acceleration for inference models",
    "area:gpu,nvidia-upgrade,fsv",
    capability="TensorRT (used across NVIDIA's RTX video stack) can materially speed our inference models on Blackwell.",
    root_cause="Models run in eager PyTorch; no TensorRT engines built.",
    dod=["Benchmark ≥1 hot model under TensorRT vs eager; integrate if it wins; assert quality parity."],
    sot="Latency benchmark + quality metric parity on real data.",
    tpo="Trigger: run model via TRT engine → Process: optimized inference → Outcome: faster, same outputs within tolerance.",
    happy="TRT engine for a hot model → ≥1.3x faster, output cosine-equal to eager within tolerance.",
    edges=["Output diverges beyond tolerance → fail.","Engine build fails on sm_120 → explicit error.","Dynamic shapes unsupported → documented limit.","Slower than eager → don't integrate.","Silent accuracy loss → broken."],
    evidence="Latency + output-equivalence report.",
)

# ===========================================================================
# F. INGEST / RENDERING / EDITING / AUDIO
# ===========================================================================

add(
    "4:2:2 10-bit hardware decode support on Blackwell (ingest + rendering)",
    "area:pipeline,area:rendering,nvidia-upgrade,fsv",
    capability="RTX Blackwell has dedicated 4:2:2 decode (up to 8K75 / 10x4K30). Supporting it speeds ingest of pro footage and avoids proxies.",
    root_cause="Pipeline assumes 4:2:0; 4:2:2 pro footage isn't hardware-decoded.",
    dod=["Detect 4:2:2 input; use NVDEC 4:2:2 path; verify color fidelity preserved."],
    sot="ffprobe of decoded frames + decode-path log (NVDEC vs CPU).",
    tpo="Trigger: ingest a 4:2:2 10-bit clip → Process: NVDEC 4:2:2 decode → Outcome: frames extracted, color preserved.",
    happy="Ingest a 4:2:2 10-bit sample → NVDEC path used, extracted frames retain 10-bit color range.",
    edges=["4:2:0 input → unchanged behaviour.","NVDEC 4:2:2 unsupported on host → explicit error/CPU note, not silent wrong colors.","8-bit upscaled to 10-bit incorrectly → fail.","Decode falls to CPU silently → surfaced.","Color shift vs source → fail."],
    evidence="ffprobe pix_fmt + decode-path log.",
    refs="NVIDIA RTX Blackwell 4:2:2 (Jun 2025 RTX AI Garage).",
)

add(
    "Editing geometry FSV: centering, fit_mode, webcam crop, no dead space",
    "area:editing,fsv",
    capability="Edits must look right: subject centered, correct fit_mode, webcam cropped tight, no black dead space.",
    root_cause="Known quality requirements (centering, fit_mode, webcam crop, no dead space) lack a geometric FSV.",
    dod=["FSV inspects rendered output geometry: subject within center band, no letterbox/dead space unless intended."],
    sot="Rendered output frames + crop/region metadata in edit_segments.",
    tpo="Trigger: create_edit + render → Process: smart crop + canvas compositing → Outcome: framed output.",
    happy="Edit a webcam clip to 1080x1920 → face bbox center within ±10% of frame center, no black bars.",
    edges=["Wide source → no pillarbox dead space.","Off-center subject → recentred.","Multi-region layout → no gaps/overlap.","fit_mode=cover vs contain honored.","Dead space present → fail."],
    evidence="Frame analysis of face bbox vs center + black-pixel-border check.",
    refs="memory: feedback_video_editing_quality.",
)

add(
    "Rendering 7-platform-profile FSV (resolution/bitrate/codec per profile)",
    "area:rendering,area:testing,fsv",
    capability="Each platform profile must emit exactly the right spec file.",
    root_cause="7 profiles exist but no test asserts each output's resolution/bitrate/codec.",
    dod=["Render the same edit through all 7 profiles; ffprobe-assert each spec."],
    sot="Output MP4 files + ffprobe metadata + renders table.",
    tpo="Trigger: render(profile) → Process: FFmpeg+NVENC → Outcome: spec-correct MP4.",
    happy="TikTok→1080x1920, YouTube4K→2160p, etc.; codec h264/h265 + bitrate within profile bounds for all 7.",
    edges=["Wrong resolution → fail.","Bitrate out of range → fail.","NVENC unavailable → explicit error, not silent CPU + wrong perf.","Audio missing in output → fail.","renders row not written → provenance gap."],
    evidence="ffprobe table for all 7 outputs + renders rows.",
)

add(
    "Audio engine FSV: ACE-Step music, MIDI, SFX, cleanup, mixing (real outputs)",
    "area:audio,area:testing,fsv",
    capability="AI music, MIDI presets, DSP SFX, cleanup, and speech-aware mixing must each produce correct audio.",
    root_cause="Audio modules largely untested; outputs not verified.",
    dod=["Per-module FSV: music gen produces non-silent audio of requested duration/BPM; SFX types distinct; cleanup reduces noise; mix ducks under speech."],
    sot="Output WAV files + audio_assets table + measured loudness/spectrum.",
    tpo="Trigger: generate_music/compose_midi/generate_sfx/audio_cleanup → Process: synthesis/DSP → Outcome: audio file + DB row.",
    happy="generate_music(120 BPM, upbeat, 10s) → 10s non-silent file, detected tempo ≈120; cleanup lowers noise floor measurably.",
    edges=["0-duration request → error, not empty file.","Cleanup on already-clean audio → no damage.","SFX 'whoosh' vs 'impact' spectrally identical → broken.","Music file silent → fail.","Ducking absent under speech → fail."],
    evidence="ffprobe duration + tempo/loudness analysis + audio_assets rows.",
)

add(
    "Auto-music mood→preset selection FSV",
    "area:audio,fsv",
    capability="auto_music must analyze an edit's mood and pick a fitting MIDI preset or trigger AI gen.",
    root_cause="music_planner mood→preset mapping unverified.",
    dod=["Given edits of known mood, assert selected preset matches expectation."],
    sot="Selected preset/asset in audio_assets + planner decision log.",
    tpo="Trigger: auto_music(edit) → Process: mood analysis → Outcome: preset/AI choice persisted.",
    happy="Calm reflective edit → ambient/corporate preset (not 'upbeat'); energetic edit → upbeat.",
    edges=["Neutral edit → sensible default, logged.","Empty edit → error, not random pick.","Mood misread (calm→aggressive) → fail.","No preset persisted → fail.","Same preset for opposite moods → broken."],
    evidence="Mood→preset decision log + audio_assets row.",
)

# ===========================================================================
# G. DASHBOARD / PROVENANCE
# ===========================================================================

add(
    "Dashboard UI end-to-end FSV (routes render real data; Playwright/Synapse)",
    "area:dashboard,area:testing,fsv",
    capability="The web dashboard must actually display real project/credit/provenance data, not just have backend routes.",
    root_cause="Flask/FastAPI routes exist but the UI is unvalidated end-to-end.",
    dod=["Browser-driven test loads each major view against a real project and asserts real values render."],
    sot="Rendered DOM values vs the underlying DB rows.",
    tpo="Trigger: open dashboard → Process: route→DB→template → Outcome: real data shown.",
    happy="With a real ingested project, the projects view shows that project; credits view matches license.db balance; provenance view shows the chain.",
    edges=["No projects → empty state, not a 500.","Tampered provenance → verify view flags it.","Credits balance mismatch DOM vs DB → fail.","Auth required route open without token → fail.","Blank page / JS error → fail."],
    evidence="Screenshots + DOM-vs-DB value diffs.",
)

add(
    "Provenance tools half-migration: decide internal-only vs re-expose; fix dashboard refs; chain-verify FSV",
    "area:dashboard,area:packaging,root-cause,fsv",
    capability="Provenance must be coherent: either internal-only (and the dashboard doesn't reference removed MCP tools) or re-exposed deliberately.",
    root_cause="Provenance functions exist but PROVENANCE_TOOL_DEFINITIONS=[] (unregistered) while dashboard routes still reference them — a half-migration.",
    dod=["Decision recorded; dashboard references reconciled; chain verification proven to detect tampering."],
    sot="provenance chain table + dashboard provenance route response.",
    tpo="Trigger: verify(project) → Process: SHA-256 chain walk → Outcome: valid/invalid verdict.",
    happy="Intact project → verify=valid; tamper one row's hash → verify=invalid pinpointing the break.",
    edges=["Dashboard calls a removed tool → 500 → fail (must be reconciled).","Empty chain → handled.","Reordered records → detected.","Verify says valid on tampered data → broken.","Genesis record missing → explicit error."],
    evidence="Verify output on intact vs tampered chain + dashboard route status.",
)

# NOTE: Billing/credits/license/D1-sync issues are intentionally OUT OF SCOPE.
# ClipCannon is an open project with no billing system (owner decision).
# (Issues #55 and #56 were closed for this reason.)

# ===========================================================================
# H. TESTING INFRA / PACKAGING / HYGIENE
# ===========================================================================

add(
    "Build the reusable Full State Verification harness (read-source-of-truth helpers)",
    "area:testing,fsv",
    capability="Every other FSV issue needs the same primitives: read a sqlite table, read a vec_* table with dim/variance, ffprobe a media file, inspect frames, diff DOM vs DB. Provide them once.",
    root_cause="No shared FSV utility exists, so each test would re-implement source-of-truth reads (and risk relying on return values).",
    dod=[
        "Helper lib: assert_table_rows, assert_vector_store(dim,variance,count), assert_media(ffprobe spec), assert_no_nan, before/after state snapshot+diff.",
        "Used by ≥3 other FSV issues.",
        "No mock backends — reads real artifacts.",
    ],
    sot="The helpers themselves, exercised against a real fixture project's DB + files.",
    tpo="Trigger: call helper on a real project → Process: direct read of DB/file → Outcome: pass/fail with the actual observed values printed.",
    happy="Run helpers on a real ingested project → they print actual row counts/dims/variances and pass; flipping an expectation makes them fail.",
    edges=["DB locked → clear error, not hang.","File missing → explicit path error.","Vector dim mismatch → fail with observed vs expected.","Helper passes when artifact absent → broken (must verify existence).","NaN slips through assert_no_nan → broken."],
    evidence="Helper output logs printing real observed state on a fixture.",
)

add(
    "CI gate: forbid mocks in tests, detect broken-state-masking, require real fixtures",
    "area:testing,fsv",
    capability="Institutionalize the no-mock / no-cover-up rule so it can't regress.",
    root_cause="Nothing prevents a future test from mocking the DB/model or passing while the system is broken.",
    dod=["CI step flags unittest.mock/MagicMock in non-allowlisted tests; requires real fixtures; runs the FSV suite.",],
    sot="CI run result + the lint report listing any mock usage.",
    tpo="Trigger: CI on PR → Process: scan tests + run FSV → Outcome: fail if mocks/broken-state detected.",
    happy="A PR that adds a mocked DB test → CI fails citing the file/line.",
    edges=["Legit allowlisted mock (external paid API) → permitted with justification.","Test asserts nothing → flagged.","FSV suite skipped → CI fails.","Mock hidden via alias import → still caught.","Green CI on a known-broken stage → broken gate."],
    evidence="CI log showing the gate firing on a planted violation.",
)

add(
    "Enforce no-runtime-model-downloads (error if a model is not pre-cached)",
    "area:packaging,root-cause,fsv",
    capability="Models must be pre-cached; a missing model must error with the exact path, never silently download at runtime (slow, non-deterministic, offline-breaking).",
    root_cause="Some loaders may auto-download from HF/torch.hub on first use.",
    dod=["Loaders check local cache and RAISE with the missing path + how to fetch; downloads disabled at runtime (HF_HUB_OFFLINE etc.).",],
    sot="Loader behaviour with cache present vs absent + network-call audit.",
    tpo="Trigger: load model with empty cache → Process: cache check → Outcome: explicit error naming the model + path.",
    happy="With offline mode + missing model → load raises naming the model and expected path; with cache present → loads, zero network calls.",
    edges=["Network available but model missing → still errors (no silent download).","Partial/corrupt cache → detected, not used.","Env offline flag unset → still must not auto-download.","Silent download occurs → broken.","Error lacks the path → not done.",],
    evidence="Loader error message + a netstat/HF-offline audit showing no download.",
    refs="memory: feedback_no_model_downloads.",
)

add(
    "Repo hygiene: remove stray ~/ and ./~ dirs, enforce file-org rules, .gitignore audit",
    "area:packaging",
    capability="Keep the repo clean per CLAUDE.md (no root-folder dumping; src/tests/docs/scripts layout).",
    root_cause="Top level contains stray `~`, `./~/.clipcannon`, temp/ and tmp/ artifacts.",
    dod=["Remove/relocate stray dirs; verify nothing important inside first; .gitignore covers artifacts; no working files in root.",],
    sot="Repo tree + git status.",
    tpo="Trigger: cleanup → Process: inspect→relocate/remove → Outcome: clean tree.",
    happy="`ls` shows no `~`/`./~`/stray temp dirs; git status clean; tests still pass.",
    edges=["A stray dir holds real data → relocate, not delete.","Deleting something tracked+needed → caught by inspection first.",".gitignore misses a new artifact dir → fail.","Root contains a new .md/test file → violates CLAUDE.md.","Cleanup breaks an import path → tests catch it."],
    evidence="Before/after tree + passing test run.",
)

add(
    "MCP server packaging FSV: clipcannon launches via absolute venv path (Local scope)",
    "area:packaging,fsv",
    capability="The MCP server must start reliably for the user; bare `clipcannon` ENOENTs — needs absolute venv path in Local scope.",
    root_cause="Bare command in MCP config fails to launch; absolute venv path required (per prior finding).",
    dod=[".mcp.json / claude mcp add uses the absolute venv binary path; server starts and lists tools.",],
    sot=".mcp.json contents + a live tool listing from the running server.",
    tpo="Trigger: MCP client starts server → Process: spawn via absolute path → Outcome: server up, tools enumerated.",
    happy="Configured with absolute venv path → `claude mcp` shows clipcannon connected and tools list returns ~54 tools.",
    edges=["Bare command → reproduce ENOENT → prove absolute path fixes it.","Wrong venv path → explicit error.","Server starts but 0 tools → broken.","Stale config after code change → restart documented (server must be restarted to pick up changes).","Works in one scope only → document scope."],
    evidence="Tool-list output from the running server + the config snippet.",
    refs="memory: feedback_clipcannon_mcp_launch, feedback_mcp_restart.",
)

add(
    "Update README/whitepaper counts to match reality (tools, stages, instruments)",
    "area:packaging,paper",
    capability="Docs must match the code so the project is accurately represented (tool count, stage count, instrument count, model list).",
    root_cause="README says 54 tools / 23 stages / 5 embeddings while code shows ~54-61 tools, more stages, and 7 instruments (paper) — drift causes confusion and undermines the paper's N=7 claim.",
    dod=["Reconcile counts to a single source of truth; README + whitepaper + paper agree on N (7, soon 8) instruments and tool/stage counts.",],
    sot="Generated count (from tool registry + pipeline registry) vs the numbers printed in docs.",
    tpo="Trigger: count script → Process: introspect registries → Outcome: authoritative counts.",
    happy="A count script prints N_tools, N_stages, N_instruments; README/whitepaper match those exactly.",
    edges=["Doc says 54, code has 61 → fail until reconciled.","Instrument count != paper's N → fail.","Hidden/disabled tools miscounted → define counting rule.","Counts drift again later → count script in CI.","Whitepaper contradicts the AGI paper → fail."],
    evidence="Count-script output diffed against doc strings.",
)


# ---------------------------------------------------------------------------
# FSV backfill comments for EXISTING open issues (avoid duplicates)
# ---------------------------------------------------------------------------
EXISTING_FSV = {
    1: (  # uv sync dependency resolution failures
        "## Full State Verification (added)\n"
        "**Source of Truth:** a clean venv after `uv sync` + `uv.lock` + actual import of every extra.\n"
        "**Trigger→Process→Outcome:** `uv sync --extra ml --extra phase2` → resolver → all deps importable.\n"
        "**Happy path:** fresh checkout + `uv sync` exits 0; `python -c 'import torch,transformers,whisperx,demucs,cupy'` succeeds; CUDA build of torch matches sm_120.\n"
        "**5 edge cases that mean broken:**\n"
        "1. Resolver picks a CPU-only torch on a CUDA box.\n"
        "2. mmcv pin reintroduces the sm_120 conflict.\n"
        "3. `uv sync` succeeds but an extra fails to import.\n"
        "4. Lockfile drifts from pyproject (out of sync).\n"
        "5. Transitive pin conflict resolved by silently dropping a needed package.\n"
        "**Evidence:** `uv sync` log + import smoke-test output + `python -c 'import torch;print(torch.version.cuda)'`.\n"
        "**Constraints:** no workarounds — root-cause the resolver conflict; no mock; error loudly on missing CUDA wheels."
    ),
    2: (  # add clipcannon MCP to .mcp.json
        "## Full State Verification (added)\n"
        "**Source of Truth:** `.mcp.json` + a live tool listing from the started server.\n"
        "**Trigger→Process→Outcome:** MCP client launches server via absolute venv path → tools enumerated.\n"
        "**Happy path:** server connects; ~54 tools listed; a no-op tool (config_list) returns GPU info.\n"
        "**5 edge cases that mean broken:** (1) bare command ENOENT; (2) wrong venv path; (3) 0 tools listed; (4) stale after code change (needs restart); (5) works only in one scope.\n"
        "**Evidence:** tool-list output + config snippet. See memory feedback_clipcannon_mcp_launch / feedback_mcp_restart."
    ),
    5: (  # EPIC Santa identity-locked generation
        "## Full State Verification (added)\n"
        "**Source of Truth:** rendered clone MP4 + per-frame ArcFace identity score log + provenance record.\n"
        "**Trigger→Process→Outcome:** clone_video(Santa, script) → EchoMimic/Audio2Face → identity-locked MP4.\n"
        "**Happy path:** Santa renders end-to-end; mean ArcFace identity > calibrated threshold; lips track audio envelope (corr>0.5); no mmcv import error.\n"
        "**5 edge cases that mean broken:** (1) interviewer/wrong face scores above threshold; (2) identity drifts across frames; (3) mmcv/sm_120 crash; (4) lips static during speech; (5) output 'passes' with no face detected.\n"
        "**Evidence:** identity-score-per-frame plot + output ffprobe + envelope correlation. No fallbacks; real reference only."
    ),
    10: (  # Prompt<->reference identity-consistency validator
        "## Full State Verification (added)\n"
        "**Source of Truth:** validator score per (prompt, reference, generated frame) + decision log.\n"
        "**Trigger→Process→Outcome:** generate → ArcFace crop vs reference → consistency verdict.\n"
        "**Happy path:** matching identity → consistent=true; swap reference to a different person → consistent=false.\n"
        "**5 edge cases that mean broken:** (1) different person scored consistent; (2) no face → auto-pass; (3) global SigLIP used instead of ArcFace crop; (4) threshold uncalibrated; (5) validator never rejects anything.\n"
        "**Evidence:** positive vs negative score table. See memory feedback_identity_verification_arcface."
    ),
    14: (  # Document reproduced success: results + provenance
        "## Full State Verification (added)\n"
        "**Source of Truth:** the committed results artifacts + provenance chain hashes they reference.\n"
        "**Trigger→Process→Outcome:** reproduce run → artifacts written → provenance chain verifies.\n"
        "**Happy path:** documented metrics (e.g. WavLM 0.961, DNSMOS 3.93) are reproducible from the stored inputs; provenance verify=valid.\n"
        "**5 edge cases that mean broken:** (1) numbers in doc not reproducible from artifacts; (2) provenance chain invalid; (3) inputs missing; (4) metric computed with a different encoder than stated; (5) doc cites a run with no artifacts.\n"
        "**Evidence:** re-run metric output matching the documented values + provenance verify log."
    ),
    15: (  # Semantic mouth-occlusion detection via face-parsing
        "## Full State Verification (added)\n"
        "**Source of Truth:** per-frame occlusion mask/flag + the frames it gates.\n"
        "**Trigger→Process→Outcome:** frame → face-parsing → mouth-occlusion flag → gates lip-sync compositing.\n"
        "**Happy path:** hand-over-mouth frame flagged occluded; open clear mouth flagged not-occluded.\n"
        "**5 edge cases that mean broken:** (1) occluded mouth not detected → garbled lips composited; (2) clear mouth falsely flagged; (3) profile/extreme angle misparsed; (4) flag computed but not used to gate; (5) parsing crashes on no-face frame.\n"
        "**Evidence:** labelled frame set → predicted vs actual occlusion flags. Follow-up to #8."
    ),
}


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> None:
    create = "--create" in sys.argv
    print(f"Total new issues to create: {len(I)}")
    for i, (t, lbl, _) in enumerate(I, 1):
        print(f"  {i:2d}. [{lbl.split(',')[0]}] {t}")
    print(f"FSV backfill comments on existing issues: {sorted(EXISTING_FSV)}")
    if not create:
        print("\nDry run. Re-run with --create to apply.")
        return

    # 1. labels
    for name, color, desc in LABELS:
        rc, out = run(["gh", "label", "create", name, "--color", color,
                       "--description", desc, "--force", "--repo", REPO])
        print(f"label {name}: {'ok' if rc == 0 else out}")

    # 2. issues
    created = []
    for t, lbl, b in I:
        rc, out = run(["gh", "issue", "create", "--repo", REPO,
                       "--title", t, "--label", lbl, "--body", b])
        print(f"{'OK ' if rc == 0 else 'ERR'} {t} -> {out.splitlines()[-1] if out else ''}")
        if rc == 0:
            created.append(out.splitlines()[-1])

    # 3. backfill FSV comments
    for num, comment in EXISTING_FSV.items():
        rc, out = run(["gh", "issue", "comment", str(num), "--repo", REPO, "--body", comment])
        print(f"comment #{num}: {'ok' if rc == 0 else out}")

    print(f"\nCreated {len(created)} issues.")
    for u in created:
        print(u)


if __name__ == "__main__":
    main()
