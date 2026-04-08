#!/usr/bin/env python3
"""Extract viseme lookup table from Santa's training video data.

Connects to ClipCannon DB for proj_2ea7221d, loads all transcript words
with timestamps, loads FLAME expression params, builds the viseme lookup
table, and saves it for inference.

Usage:
    python scripts/extract_visemes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from phoenix.video.viseme_map import VISEME_MAP, VisemeExtractor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH = Path.home() / ".clipcannon" / "projects" / "proj_2ea7221d" / "analysis.db"
FLAME_PATH = Path.home() / ".clipcannon" / "models" / "santa" / "flame_params.npz"
OUTPUT_PATH = Path.home() / ".clipcannon" / "models" / "santa" / "viseme_table.npz"


def main() -> None:
    print("=" * 60)
    print("  Viseme Extraction -- Santa Training Video")
    print("=" * 60)
    print()

    # Validate paths
    for label, p in [("DB", DB_PATH), ("FLAME", FLAME_PATH)]:
        if not p.exists():
            print(f"ERROR: {label} not found at {p}")
            sys.exit(1)
        print(f"  {label}: {p}")

    print(f"  Output: {OUTPUT_PATH}")
    print()

    # Initialize extractor
    t0 = time.perf_counter()
    extractor = VisemeExtractor(db_path=DB_PATH, flame_params_path=FLAME_PATH)
    t_init = time.perf_counter() - t0
    print(f"Extractor initialized in {t_init:.2f}s")
    print(f"  Words loaded: {len(extractor._words)}")
    print(f"  FLAME frames: {extractor._expression.shape[0]}")
    print(f"  FLAME timestamp range: {extractor._timestamps[0]:.1f}s - {extractor._timestamps[-1]:.1f}s")
    print()

    # Extract viseme table
    print("Extracting viseme table...")
    t0 = time.perf_counter()
    table = extractor.extract_viseme_table()
    t_extract = time.perf_counter() - t0
    print(f"Extraction complete in {t_extract:.2f}s")
    print()

    # Statistics
    stats = extractor.get_statistics()
    print("-" * 60)
    print("  EXTRACTION STATISTICS")
    print("-" * 60)
    print(f"  Total words processed:   {stats['total_words']}")
    print(f"  Total viseme samples:    {stats['total_samples']}")
    print(f"  Visemes covered:         {stats['visemes_covered']}/{stats['visemes_total']}")
    print(f"  Coverage:                {stats['coverage_pct']:.1f}%")
    print()

    # Per-viseme breakdown
    print(f"  {'Viseme':<8} {'ID':<4} {'Samples':>8}  {'Params (first 5)'}")
    print(f"  {'------':<8} {'--':<4} {'-------':>8}  {'-' * 40}")
    for vis_name in VISEME_MAP:
        vis_id = VISEME_MAP[vis_name]
        count = stats["per_viseme"].get(vis_name, 0)
        params = table[vis_name]
        param_str = ", ".join(f"{v:+.3f}" for v in params[:5])
        marker = " " if count > 0 else " (!) EMPTY"
        print(f"  {vis_name:<8} {vis_id:<4} {count:>8}  [{param_str}]{marker}")

    print()

    # Save
    print(f"Saving viseme table to {OUTPUT_PATH}...")
    extractor.save_table(OUTPUT_PATH)
    file_size = OUTPUT_PATH.stat().st_size
    print(f"Saved ({file_size:,} bytes)")
    print()

    # Verify save/load roundtrip
    loaded = VisemeExtractor.load_table(OUTPUT_PATH)
    for vis_name in VISEME_MAP:
        if vis_name in loaded and vis_name in table:
            assert np.allclose(loaded[vis_name], table[vis_name], atol=1e-6), \
                f"Roundtrip mismatch for {vis_name}"
    print("Save/load roundtrip verified OK")
    print()

    # Demo: text to viseme sequence
    demo_text = "Ho ho ho, Merry Christmas everyone"
    demo_dur = 3.0
    demo_fps = 25
    seq = extractor.text_to_viseme_sequence(demo_text, demo_dur, demo_fps)
    print(f"Demo: \"{demo_text}\" ({demo_dur}s @ {demo_fps}fps)")
    print(f"  Total frames: {len(seq)}")
    # Show first 20 frames
    for frame_idx, vis, params in seq[:20]:
        print(f"    frame {frame_idx:3d} t={frame_idx/demo_fps:.3f}s  {vis:<6} [{params[0]:+.2f}, {params[1]:+.2f}, {params[2]:+.2f} ...]")
    if len(seq) > 20:
        print(f"    ... ({len(seq) - 20} more frames)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
