#!/usr/bin/env python3
"""Curated training data pipeline for EchoMimicV3 LoRA v4 video clip training.

Uses ClipCannon's 23-stage analysis labels to create high-quality, categorized
training clips that distinguish speaking from breathing, filter out crying
segments, and include emotion-specific prompts.

v4 KEY CHANGE: Extracts 2-second VIDEO CLIPS (49 frames at 25fps) instead of
single frames. The model will learn HOW the mouth moves over time, not just
what the face looks like in a frozen moment.

Categories:
  - SPEAKING:   High energy + transcript = active mouth movement
  - BREATHING:  Low energy + no transcript = lips closed, subtle motion
  - EMOTIONAL:  High arousal/valence deviation = expressive moments
  - CLEAR_EYES: Excludes known crying segments (~800-850s)
"""

import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("curate")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ID = "proj_2ea7221d"
DB_PATH = Path(os.path.expanduser(
    "~/.clipcannon/projects/proj_2ea7221d/analysis.db"
))
SOURCE_VIDEO = Path(os.path.expanduser(
    "~/.clipcannon/projects/proj_2ea7221d/source/2026-04-03 04-23-11.mp4"
))
VOCALS_WAV = Path(os.path.expanduser(
    "~/.clipcannon/projects/proj_2ea7221d/stems/vocals.wav"
))
OUTPUT_DIR = Path(os.path.expanduser(
    "~/echomimic_v3/datasets/santa_curated_v5"
))

# v4: 2-second clips = 49 frames at 25fps. This matches EchoMimicV3's temporal
# compression ratio (49 pixel frames → 13 latent frames). The model learns
# temporal mouth movement patterns over these 2 seconds.
CLIP_DURATION_S = 2.0
CLIP_DURATION_MS = int(CLIP_DURATION_S * 1000)
VIDEO_FPS = 25
VIDEO_FRAMES = 49  # 2s * 25fps - 1 + 1 for alignment
TARGET_SR = 16000
IMG_WIDTH = 480
IMG_HEIGHT = 480

# Crying segment exclusion range (seconds)
CRYING_START_S = 800.0
CRYING_END_S = 850.0

# Base prompt shared by all clips
BASE_PROMPT = (
    "SANTA An older man dressed as Santa Claus with a white beard, "
    "round glasses, and red suit"
)

# Category-specific prompt suffixes
CATEGORY_PROMPTS = {
    "speaking": (
        ", speaking naturally, mouth moving with words, "
        "natural facial expressions and head movements, "
        "warm interview lighting. Don't blink too often. "
        "Preserve background integrity."
    ),
    "breathing": (
        ", lips gently closed, natural breathing, not speaking, "
        "subtle body movement, calm neutral expression, "
        "warm interview lighting. Don't blink too often. "
        "Preserve background integrity."
    ),
    "emotional": (
        ", {emotion_desc}, expressive facial movements, "
        "natural emotional response, warm interview lighting. "
        "Don't blink too often. Preserve background integrity."
    ),
    "clear_eyes": (
        ", clear bright focused eyes, engaged expression, "
        "speaking naturally, warm interview lighting. "
        "Don't blink too often. Preserve background integrity."
    ),
}


@dataclass
class ClipLabel:
    """Metadata for a single curated training clip."""
    clip_id: str
    category: str
    start_ms: int
    end_ms: int
    energy_mean: float = 0.0
    speaking_rate_wpm: float = 0.0
    f0_mean: float = 0.0
    emotion_label: str = "neutral"
    arousal: float = 0.0
    valence: float = 0.0
    transcript_text: str = ""
    prompt: str = ""
    has_transcript: bool = False
    in_crying_range: bool = False
    words: list = field(default_factory=list)


def load_analysis_data(db_path: Path, project_id: str) -> dict:
    """Load all analysis tables from the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    data = {}

    # Prosody segments
    rows = conn.execute(
        """SELECT start_ms, end_ms, f0_mean, energy_mean, speaking_rate_wpm,
                  emotion_label, energy_level, has_emphasis, has_breath,
                  transcript_text, word_count
           FROM prosody_segments WHERE project_id=?
           ORDER BY start_ms""",
        (project_id,),
    ).fetchall()
    data["prosody"] = [dict(r) for r in rows]
    log.info("Loaded %d prosody segments", len(data["prosody"]))

    # Transcript segments (with speaker_id)
    rows = conn.execute(
        """SELECT start_ms, end_ms, text, speaker_id
           FROM transcript_segments WHERE project_id=?
           ORDER BY start_ms""",
        (project_id,),
    ).fetchall()
    data["transcript"] = [dict(r) for r in rows]
    log.info("Loaded %d transcript segments", len(data["transcript"]))

    # Emotion curve (deduplicate by averaging per time window)
    rows = conn.execute(
        """SELECT start_ms, end_ms, AVG(arousal) as arousal,
                  AVG(valence) as valence, AVG(energy) as energy
           FROM emotion_curve WHERE project_id=?
           GROUP BY start_ms, end_ms
           ORDER BY start_ms""",
        (project_id,),
    ).fetchall()
    data["emotion"] = [dict(r) for r in rows]
    log.info("Loaded %d emotion windows (deduplicated)", len(data["emotion"]))

    # Speakers
    rows = conn.execute(
        """SELECT speaker_id, label, total_speaking_ms, speaking_pct
           FROM speakers WHERE project_id=?
           ORDER BY total_speaking_ms DESC""",
        (project_id,),
    ).fetchall()
    data["speakers"] = [dict(r) for r in rows]
    log.info("Loaded %d speakers", len(data["speakers"]))

    # Silence gaps (useful for breathing identification)
    rows = conn.execute(
        """SELECT start_ms, end_ms, duration_ms, type
           FROM silence_gaps WHERE project_id=?
           ORDER BY start_ms""",
        (project_id,),
    ).fetchall()
    data["silence_gaps"] = [dict(r) for r in rows]
    log.info("Loaded %d silence gaps", len(data["silence_gaps"]))

    # Scene map with face positions for face-centered cropping
    rows = conn.execute(
        """SELECT start_ms, end_ms, face_x, face_y, face_w, face_h,
                  webcam_x, webcam_y, webcam_w, webcam_h
           FROM scene_map WHERE project_id=? AND face_w > 0
           ORDER BY start_ms""",
        (project_id,),
    ).fetchall()
    data["scene_map"] = [dict(r) for r in rows]
    log.info("Loaded %d scene_map entries with face data", len(data["scene_map"]))

    # Word-level timestamps for viseme/phoneme alignment
    rows = conn.execute(
        """SELECT w.word_id, w.word, w.start_ms, w.end_ms, w.confidence,
                  w.speaker_id, s.text as segment_text
           FROM transcript_words w
           JOIN transcript_segments s ON w.segment_id = s.segment_id
           WHERE s.project_id=?
           ORDER BY w.start_ms""",
        (project_id,),
    ).fetchall()
    data["words"] = [dict(r) for r in rows]
    log.info("Loaded %d transcript words", len(data["words"]))

    conn.close()
    return data


def identify_santa_speaker(data: dict) -> Optional[int]:
    """Find Santa's speaker_id (most speaking time)."""
    if not data["speakers"]:
        log.warning("No speakers found, assuming all speech is Santa")
        return None
    santa = data["speakers"][0]
    log.info(
        "Santa identified: speaker_id=%d, label=%s, speaking=%.1f%%",
        santa["speaker_id"], santa["label"], santa["speaking_pct"],
    )
    return santa["speaker_id"]


def has_transcript_overlap(start_ms: int, end_ms: int,
                           transcript: list, santa_id: Optional[int]) -> tuple:
    """Check if a time range has overlapping transcript text from Santa."""
    overlapping_text = []
    for seg in transcript:
        # Filter for Santa's speech if speaker_id is available
        if santa_id is not None and seg["speaker_id"] != santa_id:
            continue
        # Check overlap
        seg_start = seg["start_ms"]
        seg_end = seg["end_ms"]
        overlap = min(end_ms, seg_end) - max(start_ms, seg_start)
        if overlap > 0:
            overlapping_text.append(seg["text"])
    return len(overlapping_text) > 0, " ".join(overlapping_text)


def get_emotion_at(start_ms: int, end_ms: int, emotion_data: list) -> tuple:
    """Get average arousal/valence for a time range from emotion curve."""
    arousal_vals = []
    valence_vals = []
    for em in emotion_data:
        overlap = min(end_ms, em["end_ms"]) - max(start_ms, em["start_ms"])
        if overlap > 0:
            arousal_vals.append(em["arousal"])
            valence_vals.append(em["valence"])
    if not arousal_vals:
        return 0.19, 0.507  # median defaults
    return np.mean(arousal_vals), np.mean(valence_vals)


def is_in_crying_range(start_ms: int, end_ms: int) -> bool:
    """Check if clip overlaps with known crying segments."""
    cry_start_ms = int(CRYING_START_S * 1000)
    cry_end_ms = int(CRYING_END_S * 1000)
    overlap = min(end_ms, cry_end_ms) - max(start_ms, cry_start_ms)
    return overlap > 0


def arousal_to_emotion_desc(arousal: float, valence: float) -> str:
    """Map arousal/valence to Plutchik emotion description.

    The emotion curve values are fairly compressed (arousal: 0.15-0.23,
    valence: 0.50-0.51), so we use relative thresholds.
    """
    # Compute deviations from median
    arousal_dev = arousal - 0.192  # median arousal
    valence_dev = valence - 0.507  # median valence

    if arousal_dev > 0.01 and valence_dev > 0.001:
        return "joyful and animated, bright engaged expression"
    elif arousal_dev > 0.01 and valence_dev < -0.001:
        return "passionate and intense, deeply emotional expression"
    elif arousal_dev < -0.01 and valence_dev > 0.001:
        return "serene and content, gentle warm expression"
    elif arousal_dev < -0.01 and valence_dev < -0.001:
        return "somber and reflective, thoughtful expression"
    elif arousal_dev > 0.01:
        return "animated and energetic, expressive face"
    elif arousal_dev < -0.01:
        return "calm and composed, measured expression"
    else:
        return "naturally expressive, warm emotional tone"


def compute_energy_percentiles(prosody: list) -> dict:
    """Compute energy percentiles for adaptive thresholds."""
    energies = sorted(p["energy_mean"] for p in prosody)
    n = len(energies)
    return {
        "p25": energies[int(n * 0.25)],
        "p50": energies[int(n * 0.50)],
        "p75": energies[int(n * 0.75)],
    }


def build_clip_windows(video_duration_ms: int, step_ms: int = 1000) -> list:
    """Generate overlapping 2-second windows across the video.

    Uses 1-second step for dense coverage of 2s clips, giving many candidates
    for the diversity selection pass.
    """
    windows = []
    for start in range(0, video_duration_ms - CLIP_DURATION_MS + 1, step_ms):
        windows.append((start, start + CLIP_DURATION_MS))
    return windows


def find_transcript_gaps(transcript: list, santa_id: Optional[int],
                         min_gap_ms: int = 2000) -> list:
    """Find gaps between Santa's transcript segments for breathing clips.

    Returns list of (gap_start_ms, gap_end_ms) where Santa is NOT speaking.
    Uses 2-second minimum to capture natural pauses and breathing moments.
    """
    # Filter to Santa's segments only
    santa_segs = [
        s for s in transcript
        if santa_id is None or s["speaker_id"] == santa_id
    ]
    if not santa_segs:
        return []

    santa_segs.sort(key=lambda s: s["start_ms"])
    gaps = []
    for i in range(len(santa_segs) - 1):
        gap_start = santa_segs[i]["end_ms"]
        gap_end = santa_segs[i + 1]["start_ms"]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_gap_ms:
            gaps.append((gap_start, gap_end))

    return gaps


def select_clips(data: dict) -> list:
    """Select and categorize clips using analysis data."""
    santa_id = identify_santa_speaker(data)
    prosody = data["prosody"]
    transcript = data["transcript"]
    emotion = data["emotion"]

    energy_pcts = compute_energy_percentiles(prosody)
    log.info(
        "Energy percentiles: p25=%.6f, p50=%.6f, p75=%.6f",
        energy_pcts["p25"], energy_pcts["p50"], energy_pcts["p75"],
    )

    # Compute arousal stats for emotion thresholds
    all_arousal = [em["arousal"] for em in emotion]
    all_valence = [em["valence"] for em in emotion]
    arousal_mean = np.mean(all_arousal)
    arousal_std = np.std(all_arousal)
    valence_mean = np.mean(all_valence)
    valence_std = np.std(all_valence)
    log.info(
        "Emotion stats: arousal=%.4f +/- %.4f, valence=%.4f +/- %.4f",
        arousal_mean, arousal_std, valence_mean, valence_std,
    )

    # Get video duration from last prosody/transcript segment
    video_duration_ms = max(
        max((p["end_ms"] for p in prosody), default=0),
        max((t["end_ms"] for t in transcript), default=0),
    )
    log.info("Video duration: %.1fs (%dms)", video_duration_ms / 1000, video_duration_ms)

    # Build time windows for speaking/emotional/clear_eyes
    windows = build_clip_windows(video_duration_ms)
    log.info("Generated %d candidate windows (2s step)", len(windows))

    # Find transcript gaps for breathing clips
    transcript_gaps = find_transcript_gaps(transcript, santa_id, min_gap_ms=2000)
    log.info("Found %d transcript gaps for breathing candidates", len(transcript_gaps))

    # For each window, compute features from overlapping prosody segments
    clips = {"speaking": [], "breathing": [], "emotional": [], "clear_eyes": []}

    for start_ms, end_ms in windows:
        # Skip crying range entirely for all categories
        if is_in_crying_range(start_ms, end_ms):
            continue

        # Get overlapping prosody segments
        overlap_prosody = []
        for p in prosody:
            overlap = min(end_ms, p["end_ms"]) - max(start_ms, p["start_ms"])
            if overlap > 500:  # at least 500ms overlap
                overlap_prosody.append(p)

        # Compute window-level energy and speaking rate
        if overlap_prosody:
            avg_energy = np.mean([p["energy_mean"] for p in overlap_prosody])
            avg_rate = np.mean([p["speaking_rate_wpm"] for p in overlap_prosody])
            avg_f0 = np.mean([p["f0_mean"] for p in overlap_prosody])
        else:
            avg_energy = 0.0
            avg_rate = 0.0
            avg_f0 = 0.0

        # Check transcript overlap (Santa only)
        has_text, text = has_transcript_overlap(
            start_ms, end_ms, transcript, santa_id
        )

        # Get emotion data
        arousal, valence = get_emotion_at(start_ms, end_ms, emotion)

        base_label = ClipLabel(
            clip_id="",
            category="",
            start_ms=start_ms,
            end_ms=end_ms,
            energy_mean=avg_energy,
            speaking_rate_wpm=avg_rate,
            f0_mean=avg_f0,
            arousal=arousal,
            valence=valence,
            transcript_text=text[:200] if text else "",
            has_transcript=has_text,
            in_crying_range=False,
        )

        # Category 1: SPEAKING - above-median energy + active transcript
        if (avg_energy > energy_pcts["p50"]
                and avg_rate > 150
                and has_text):
            label = ClipLabel(**{**asdict(base_label),
                                 "category": "speaking",
                                 "emotion_label": "speaking"})
            clips["speaking"].append(label)

        # Category 3: EMOTIONAL - high arousal or valence deviation
        arousal_dev = abs(arousal - arousal_mean)
        valence_dev = abs(valence - valence_mean)
        if (arousal_dev > arousal_std * 0.8 or valence_dev > valence_std * 0.8):
            emotion_desc = arousal_to_emotion_desc(arousal, valence)
            label = ClipLabel(**{**asdict(base_label),
                                 "category": "emotional",
                                 "emotion_label": emotion_desc})
            clips["emotional"].append(label)

        # Category 4: CLEAR_EYES - good quality, not crying
        if (avg_energy > energy_pcts["p25"]
                and has_text
                and not is_in_crying_range(start_ms, end_ms)):
            label = ClipLabel(**{**asdict(base_label),
                                 "category": "clear_eyes",
                                 "emotion_label": "clear_eyes"})
            clips["clear_eyes"].append(label)

    # Category 2: BREATHING - from transcript gaps (where Santa is NOT speaking)
    # Strategy: center 6-second clips in gaps. For gaps shorter than 6s,
    # we still create a clip centered on the gap (it will include some
    # speech edges, but the majority is silence/breathing which is the
    # critical training signal for "lips closed").
    for gap_start, gap_end in transcript_gaps:
        if is_in_crying_range(gap_start, gap_end):
            continue
        gap_dur = gap_end - gap_start

        if gap_dur >= CLIP_DURATION_MS:
            # Gap is long enough: place clips with 2-second steps
            for offset in range(0, gap_dur - CLIP_DURATION_MS + 1, 2000):
                clip_start = gap_start + offset
                clip_end = clip_start + CLIP_DURATION_MS
                arousal, valence = get_emotion_at(clip_start, clip_end, emotion)
                label = ClipLabel(
                    clip_id="",
                    category="breathing",
                    start_ms=clip_start,
                    end_ms=clip_end,
                    energy_mean=0.0,
                    speaking_rate_wpm=0.0,
                    f0_mean=0.0,
                    arousal=arousal,
                    valence=valence,
                    emotion_label="breathing",
                    transcript_text="",
                    has_transcript=False,
                    in_crying_range=False,
                )
                clips["breathing"].append(label)
        elif gap_dur >= 1500:
            # Gap is 1.5-2 seconds: center a 2-second clip on the gap.
            # The clip may extend into speech edges but the gap center
            # teaches the model that low-energy = lips closed.
            gap_center = (gap_start + gap_end) // 2
            clip_start = max(0, gap_center - CLIP_DURATION_MS // 2)
            clip_end = clip_start + CLIP_DURATION_MS
            arousal, valence = get_emotion_at(clip_start, clip_end, emotion)
            label = ClipLabel(
                clip_id="",
                category="breathing",
                start_ms=clip_start,
                end_ms=clip_end,
                energy_mean=0.0,
                speaking_rate_wpm=0.0,
                f0_mean=0.0,
                arousal=arousal,
                valence=valence,
                emotion_label="breathing",
                transcript_text="",
                has_transcript=False,
                in_crying_range=False,
            )
            clips["breathing"].append(label)

    # Additional breathing candidates from low-energy prosody segments
    # with breath flag (natural breathing moments even within speech regions)
    breathing_times = set()  # Track to avoid duplicates
    for br_clip in clips["breathing"]:
        breathing_times.add(br_clip.start_ms)

    for p in prosody:
        if (p["has_breath"] and p["energy_mean"] < energy_pcts["p25"]
                and p["speaking_rate_wpm"] < 120):
            seg_center = (p["start_ms"] + p["end_ms"]) // 2
            clip_start = max(0, seg_center - CLIP_DURATION_MS // 2)
            clip_end = clip_start + CLIP_DURATION_MS
            if is_in_crying_range(clip_start, clip_end):
                continue
            # Skip if too close to existing breathing clip
            if any(abs(clip_start - t) < 3000 for t in breathing_times):
                continue
            arousal, valence = get_emotion_at(clip_start, clip_end, emotion)
            label = ClipLabel(
                clip_id="",
                category="breathing",
                start_ms=clip_start,
                end_ms=clip_end,
                energy_mean=p["energy_mean"],
                speaking_rate_wpm=p["speaking_rate_wpm"],
                f0_mean=p["f0_mean"],
                arousal=arousal,
                valence=valence,
                emotion_label="breathing",
                transcript_text="",
                has_transcript=False,
                in_crying_range=False,
            )
            clips["breathing"].append(label)
            breathing_times.add(clip_start)

    log.info("Category candidates: speaking=%d, breathing=%d, emotional=%d, clear_eyes=%d",
             len(clips["speaking"]), len(clips["breathing"]),
             len(clips["emotional"]), len(clips["clear_eyes"]))

    return clips


def select_diverse_subset(clips: list, target_count: int,
                          min_gap_ms: int = 2000) -> list:
    """Select a diverse subset of clips with minimum temporal gap.

    Uses a two-pass approach:
    1. Strict gap enforcement, ranking by energy
    2. Relaxed gap (no exact overlap) to fill remaining quota
    """
    if len(clips) <= target_count:
        return clips

    # Sort by energy (descending) to prefer high-quality clips
    ranked = sorted(clips, key=lambda c: c.energy_mean, reverse=True)

    selected = []
    used_ranges = []

    for clip in ranked:
        if len(selected) >= target_count:
            break
        # Check minimum gap from already-selected clips
        too_close = False
        for used_start, used_end in used_ranges:
            overlap = min(clip.end_ms, used_end) - max(clip.start_ms, used_start)
            if overlap > -min_gap_ms:  # negative overlap = gap
                too_close = True
                break
        if not too_close:
            selected.append(clip)
            used_ranges.append((clip.start_ms, clip.end_ms))

    # Pass 2: relax to 1-second non-overlap
    if len(selected) < target_count:
        remaining = [c for c in ranked if c not in selected]
        for clip in remaining:
            if len(selected) >= target_count:
                break
            exact_overlap = any(
                min(clip.end_ms, u[1]) - max(clip.start_ms, u[0]) > 1000
                for u in used_ranges
            )
            if not exact_overlap:
                selected.append(clip)
                used_ranges.append((clip.start_ms, clip.end_ms))

    # Pass 3: if still short, accept any non-duplicate
    if len(selected) < target_count:
        remaining = [c for c in ranked if c not in selected]
        for clip in remaining:
            if len(selected) >= target_count:
                break
            exact_dup = any(
                clip.start_ms == u[0] and clip.end_ms == u[1]
                for u in used_ranges
            )
            if not exact_dup:
                selected.append(clip)
                used_ranges.append((clip.start_ms, clip.end_ms))

    return sorted(selected, key=lambda c: c.start_ms)


def get_face_crop_for_time(start_ms: int, end_ms: int, scene_map: list,
                           src_w: int = 2560, src_h: int = 1440,
                           face_scale: float = 2.2) -> Optional[tuple]:
    """Get a face-centered square crop region from scene_map data.

    Returns (crop_x, crop_y, crop_size) or None if no face found.
    The crop is a square centered on the face with `face_scale`x the face
    size for head+shoulders context, then scaled to target resolution.
    """
    # Find scene_map entries overlapping this time range
    best_face = None
    for scene in scene_map:
        overlap = min(end_ms, scene["end_ms"]) - max(start_ms, scene["start_ms"])
        if overlap > 0 and scene.get("face_w", 0) > 0:
            # Prefer larger faces (closer shots)
            if best_face is None or scene["face_w"] > best_face["face_w"]:
                best_face = scene

    if best_face is None:
        return None

    fx = best_face["face_x"]
    fy = best_face["face_y"]
    fw = best_face["face_w"]
    fh = best_face["face_h"]
    cx = fx + fw // 2
    cy = fy + fh // 2

    # Square crop: face_scale * face size, clamped to source
    crop_sz = min(int(max(fw, fh) * face_scale), src_h, src_w)
    x1 = max(0, cx - crop_sz // 2)
    y1 = max(0, cy - crop_sz // 2)
    x1 = min(x1, src_w - crop_sz)
    y1 = min(y1, src_h - crop_sz)

    return (x1, y1, crop_sz)


def extract_frame(video_path: Path, time_s: float, output_path: Path,
                  width: int = IMG_WIDTH, height: int = IMG_HEIGHT,
                  crop: Optional[tuple] = None) -> bool:
    """Extract a single frame, optionally with face-centered crop."""
    if crop:
        cx, cy, csz = crop
        vf = f"crop={csz}:{csz}:{cx}:{cy},scale={width}:{height}"
    else:
        vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
    cmd = [
        "ffmpeg", "-y", "-ss", f"{time_s:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", vf,
        "-q:v", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0


def extract_video_clip(video_path: Path, start_s: float, duration_s: float,
                       output_path: Path, fps: int = VIDEO_FPS,
                       width: int = IMG_WIDTH, height: int = IMG_HEIGHT,
                       crop: Optional[tuple] = None) -> bool:
    """Extract a short video clip with face-centered crop.

    Produces exactly `duration_s * fps` frames for VAE encoding.
    Uses face crop to eliminate black bars and maximize face detail.
    """
    if crop:
        cx, cy, csz = crop
        vf = f"fps={fps},crop={csz}:{csz}:{cx}:{cy},scale={width}:{height}"
    else:
        vf = (f"fps={fps},"
              f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "15",
        "-preset", "fast",
        "-an",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


def extract_audio_clip(vocals_path: Path, start_s: float, duration_s: float,
                       output_path: Path, target_sr: int = TARGET_SR) -> bool:
    """Extract audio clip from vocals, downmix to mono 16kHz."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-i", str(vocals_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0


def get_words_in_range(words: list, start_ms: int, end_ms: int,
                       santa_id: Optional[int] = None) -> list:
    """Get word-level transcript data for a time range."""
    result = []
    for w in words:
        if santa_id is not None and w.get("speaker_id") != santa_id:
            continue
        if w["start_ms"] >= start_ms and w["end_ms"] <= end_ms:
            result.append({
                "word": w["word"],
                "start_ms": w["start_ms"] - start_ms,  # relative to clip start
                "end_ms": w["end_ms"] - start_ms,
                "confidence": w.get("confidence", 0.0),
            })
    return result


def build_prompt(category: str, emotion_label: str = "") -> str:
    """Build the full prompt string for a clip."""
    suffix_template = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["speaking"])
    if category == "emotional" and emotion_label:
        suffix = suffix_template.format(emotion_desc=emotion_label)
    else:
        suffix = suffix_template
    return BASE_PROMPT + suffix


def curate_training_data():
    """Main pipeline: load data, select clips, extract media, save labels."""
    log.info("=" * 70)
    log.info("CURATED TRAINING DATA PIPELINE")
    log.info("=" * 70)

    # Validate paths
    if not DB_PATH.exists():
        log.error("Database not found: %s", DB_PATH)
        sys.exit(1)
    if not SOURCE_VIDEO.exists():
        log.error("Source video not found: %s", SOURCE_VIDEO)
        sys.exit(1)
    if not VOCALS_WAV.exists():
        log.error("Vocals not found: %s", VOCALS_WAV)
        sys.exit(1)

    # 1. Load analysis data
    log.info("Loading analysis data from %s", DB_PATH)
    data = load_analysis_data(DB_PATH, PROJECT_ID)

    # 2. Select and categorize clips
    log.info("Selecting clips by category...")
    all_clips = select_clips(data)

    # 3. Select diverse subsets per category
    # v4: higher targets — 2s clips give 3x more coverage than 6s clips
    targets = {"speaking": 150, "breathing": 50, "emotional": 50, "clear_eyes": 50}
    final_clips = []

    for category, target in targets.items():
        candidates = all_clips[category]
        if not candidates:
            log.warning("No candidates for category '%s'", category)
            continue
        selected = select_diverse_subset(candidates, target)
        log.info("Selected %d/%d %s clips (from %d candidates)",
                 len(selected), target, category, len(candidates))
        final_clips.extend(selected)

    # Sort all clips by time and assign IDs
    final_clips.sort(key=lambda c: (c.category, c.start_ms))
    for i, clip in enumerate(final_clips):
        clip.clip_id = f"{i:04d}"
        clip.prompt = build_prompt(clip.category, clip.emotion_label)

    log.info("Total curated clips: %d", len(final_clips))

    # 4. Create output directories
    dirs = {
        "clips": OUTPUT_DIR / "clips",     # v4: 2-second video clips
        "imgs": OUTPUT_DIR / "imgs",       # reference frames (first frame)
        "audios": OUTPUT_DIR / "audios",   # 2-second audio clips
        "prompts": OUTPUT_DIR / "prompts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Get word-level data and Santa's speaker ID for per-clip words
    santa_id = identify_santa_speaker(data)
    scene_map = data.get("scene_map", [])

    # 5. Extract media for each clip with FACE-CENTERED CROP
    # This eliminates black bars and maximizes face detail.
    log.info("Extracting video clips with face-centered crop...")
    success_count = 0
    fail_count = 0
    no_face_count = 0

    for clip in final_clips:
        clip_id = clip.clip_id
        start_s = clip.start_ms / 1000.0

        # Get face-centered crop region from scene_map
        crop = get_face_crop_for_time(clip.start_ms, clip.end_ms, scene_map)
        if crop is None:
            no_face_count += 1
            # Fallback: use median face position from all scenes
            if scene_map:
                med_fx = int(np.median([s["face_x"] for s in scene_map]))
                med_fy = int(np.median([s["face_y"] for s in scene_map]))
                med_fw = int(np.median([s["face_w"] for s in scene_map]))
                med_fh = int(np.median([s["face_h"] for s in scene_map]))
                cx = med_fx + med_fw // 2
                cy = med_fy + med_fh // 2
                crop_sz = min(int(max(med_fw, med_fh) * 2.2), 1440)
                x1 = max(0, min(cx - crop_sz // 2, 2560 - crop_sz))
                y1 = max(0, min(cy - crop_sz // 2, 1440 - crop_sz))
                crop = (x1, y1, crop_sz)

        # Extract 2-second video clip with face crop (49 frames at 25fps)
        clip_path = dirs["clips"] / f"{clip_id}.mp4"
        if not extract_video_clip(SOURCE_VIDEO, start_s, CLIP_DURATION_S,
                                  clip_path, crop=crop):
            log.warning("Failed to extract video clip %s at %.1fs",
                        clip_id, start_s)
            fail_count += 1
            continue

        # Extract reference frame (first frame) with same crop
        img_path = dirs["imgs"] / f"{clip_id}.jpg"
        if not extract_frame(SOURCE_VIDEO, start_s, img_path, crop=crop):
            log.warning("Failed to extract frame for clip %s at %.1fs",
                        clip_id, start_s)
            clip_path.unlink(missing_ok=True)
            fail_count += 1
            continue

        # Extract audio clip from vocals (2 seconds)
        audio_path = dirs["audios"] / f"{clip_id}.wav"
        if not extract_audio_clip(VOCALS_WAV, start_s, CLIP_DURATION_S, audio_path):
            log.warning("Failed to extract audio for clip %s", clip_id)
            clip_path.unlink(missing_ok=True)
            img_path.unlink(missing_ok=True)
            fail_count += 1
            continue

        # Write prompt
        prompt_path = dirs["prompts"] / f"{clip_id}.txt"
        prompt_path.write_text(clip.prompt, encoding="utf-8")

        # Attach word-level data to clip for labels.json
        clip.words = get_words_in_range(
            data["words"], clip.start_ms, clip.end_ms, santa_id
        )

        success_count += 1
        if (success_count) % 20 == 0:
            log.info("  Extracted %d/%d clips...", success_count, len(final_clips))

    if no_face_count > 0:
        log.info("  Used fallback crop for %d clips (no face in scene_map)", no_face_count)

    log.info("Extraction complete: %d success, %d failed", success_count, fail_count)

    # 6. Save labels.json
    labels = []
    for clip in final_clips:
        clip_path = dirs["clips"] / f"{clip.clip_id}.mp4"
        if not clip_path.exists():
            continue
        labels.append({
            "clip_id": clip.clip_id,
            "category": clip.category,
            "start_ms": clip.start_ms,
            "end_ms": clip.end_ms,
            "start_s": clip.start_ms / 1000.0,
            "end_s": clip.end_ms / 1000.0,
            "duration_s": CLIP_DURATION_S,
            "video_frames": VIDEO_FRAMES,
            "video_fps": VIDEO_FPS,
            "energy_mean": round(clip.energy_mean, 6),
            "speaking_rate_wpm": round(clip.speaking_rate_wpm, 1),
            "f0_mean": round(clip.f0_mean, 4),
            "arousal": round(clip.arousal, 4),
            "valence": round(clip.valence, 4),
            "emotion_label": clip.emotion_label,
            "transcript_text": clip.transcript_text,
            "has_transcript": clip.has_transcript,
            "in_crying_range": clip.in_crying_range,
            "prompt": clip.prompt,
            "words": getattr(clip, "words", []),
        })

    labels_path = OUTPUT_DIR / "labels.json"
    labels_path.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved labels.json with %d entries", len(labels))

    # 7. Print statistics
    print_statistics(labels)

    return labels


def print_statistics(labels: list):
    """Print detailed statistics about the curated dataset."""
    log.info("")
    log.info("=" * 70)
    log.info("CURATED DATASET STATISTICS")
    log.info("=" * 70)

    total = len(labels)
    log.info("Total clips: %d", total)
    log.info("Clip duration: %.1fs each", CLIP_DURATION_S)
    log.info("Total training time: %.1fs (%.1f minutes)",
             total * CLIP_DURATION_S, total * CLIP_DURATION_S / 60)

    # Per-category breakdown
    log.info("")
    log.info("--- Category Breakdown ---")
    categories = {}
    for label in labels:
        cat = label["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(label)

    for cat in ["speaking", "breathing", "emotional", "clear_eyes"]:
        cat_labels = categories.get(cat, [])
        if not cat_labels:
            log.info("  %-12s: 0 clips", cat)
            continue
        energies = [l["energy_mean"] for l in cat_labels]
        rates = [l["speaking_rate_wpm"] for l in cat_labels]
        arousals = [l["arousal"] for l in cat_labels]
        time_range_s = sum(l["end_s"] - l["start_s"] for l in cat_labels)
        log.info("  %-12s: %3d clips | energy=%.4f-%.4f | rate=%.0f-%.0f wpm | "
                 "arousal=%.3f-%.3f | %.0fs total",
                 cat, len(cat_labels),
                 min(energies), max(energies),
                 min(rates), max(rates),
                 min(arousals), max(arousals),
                 time_range_s)

    # Speaking time coverage
    log.info("")
    log.info("--- Coverage ---")
    speaking_clips = categories.get("speaking", [])
    if speaking_clips:
        min_t = min(l["start_s"] for l in speaking_clips)
        max_t = max(l["end_s"] for l in speaking_clips)
        log.info("  Speaking range: %.1fs - %.1fs", min_t, max_t)

    breathing_clips = categories.get("breathing", [])
    if breathing_clips:
        min_t = min(l["start_s"] for l in breathing_clips)
        max_t = max(l["end_s"] for l in breathing_clips)
        log.info("  Breathing range: %.1fs - %.1fs", min_t, max_t)

    # Transcript coverage
    with_transcript = sum(1 for l in labels if l["has_transcript"])
    without_transcript = total - with_transcript
    log.info("  With transcript: %d (%.0f%%)", with_transcript,
             100 * with_transcript / max(total, 1))
    log.info("  Without transcript (breathing/silence): %d (%.0f%%)",
             without_transcript, 100 * without_transcript / max(total, 1))

    # Crying exclusion
    log.info("  Crying range excluded: %.0fs - %.0fs", CRYING_START_S, CRYING_END_S)
    log.info("")
    # Word coverage
    total_words = sum(len(l.get("words", [])) for l in labels)
    log.info("  Total word-level entries: %d", total_words)
    log.info("  Video frames per clip: %d (%dfps x %.1fs)",
             VIDEO_FRAMES, VIDEO_FPS, CLIP_DURATION_S)
    log.info("Output directory: %s", OUTPUT_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    curate_training_data()
