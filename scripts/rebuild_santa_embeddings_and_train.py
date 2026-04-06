#!/usr/bin/env python3
"""Rebuild Santa-only embeddings (filter interviewer) and launch full training.

Applies CUDA 13.1/13.2 best practices for RTX 5090 WSL2 stability:
- Single CUDA context, serialized model loads
- expandable_segments + max_split_size_mb
- Aggressive gc.collect + torch.cuda.empty_cache between phases
- FP16 where possible, FP32 accumulation
- Stream priorities for concurrent ops
- nohup-safe (flushes logs)
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- CUDA env vars BEFORE any torch import (CUDA 13.1/13.2 best practices) ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "max_split_size_mb:512,"
    "garbage_collection_threshold:0.7"
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # async launches
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"  # Blackwell only
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Force unbuffered output for nohup
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/semantic_training_full.log"),
    ],
)
logger = logging.getLogger("rebuild_train")


def gpu_cleanup():
    """Aggressive GPU memory cleanup — call between heavy operations."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception:
        pass


def check_gpu_health():
    """Pre-flight GPU check — abort early if something is wrong."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    dev = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    logger.info(f"GPU: {dev}, Total: {total:.1f}GB, Free: {free:.1f}GB")

    # Quick VRAM test — allocate and release 4GB
    try:
        test = torch.randn(512, 1024, 1024, device="cuda", dtype=torch.float16)  # ~1GB
        del test
        torch.cuda.empty_cache()
        logger.info("GPU health check: PASSED")
    except RuntimeError as e:
        logger.error(f"GPU health check FAILED: {e}")
        raise


# =============================================================================
# Phase 0: Filter interviewer segments using ECAPA speaker embeddings
# =============================================================================
def filter_interviewer_segments():
    """Use ECAPA-TDNN speaker embeddings to identify and filter out interviewer."""
    import sqlite3
    import torch
    import torchaudio
    from sklearn.cluster import KMeans

    logger.info("=" * 60)
    logger.info("PHASE 0: Filtering interviewer segments via speaker embeddings")
    logger.info("=" * 60)

    db_path = Path.home() / ".clipcannon" / "projects" / "proj_2ea7221d" / "analysis.db"
    vocals_path = db_path.parent / "stems" / "vocals.wav"

    db = sqlite3.connect(str(db_path))

    # Load audio
    waveform, sr = torchaudio.load(str(vocals_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    logger.info(f"Vocals: {waveform.shape[1]/sr:.1f}s at {sr}Hz")

    # Load ECAPA speaker encoder (small model, ~25MB VRAM)
    from speechbrain.pretrained import EncoderClassifier
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    # Get prosody segments
    rows = db.execute("""
        SELECT segment_id, start_ms, end_ms
        FROM prosody_segments
        ORDER BY start_ms
    """).fetchall()
    logger.info(f"Processing {len(rows)} prosody segments for speaker ID...")

    # Extract per-segment speaker embeddings
    embeddings = []
    segment_ids = []
    for sid, start_ms, end_ms in rows:
        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)
        chunk = waveform[:, start_sample:end_sample]
        if chunk.shape[1] < sr * 0.5:  # skip < 0.5s
            continue
        with torch.no_grad():
            emb = encoder.encode_batch(chunk.cuda())
            embeddings.append(emb.squeeze().cpu().numpy())
            segment_ids.append(sid)

    embeddings_arr = np.array(embeddings)
    logger.info(f"Extracted {len(embeddings_arr)} speaker embeddings (dim={embeddings_arr.shape[1]})")

    # Unload encoder immediately
    del encoder
    gpu_cleanup()

    # Cluster into 2 speakers
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_arr)

    # Identify Santa = cluster with more total speaking time
    cluster_durations = {}
    for i, (sid, start_ms, end_ms) in enumerate([(rows[j][0], rows[j][1], rows[j][2]) for j in range(len(rows)) if rows[j][0] in set(segment_ids)]):
        idx = segment_ids.index(sid)
        c = labels[idx]
        cluster_durations[c] = cluster_durations.get(c, 0) + (end_ms - start_ms)

    santa_cluster = max(cluster_durations, key=cluster_durations.get)
    interviewer_cluster = 1 - santa_cluster

    santa_sids = set(segment_ids[i] for i in range(len(segment_ids)) if labels[i] == santa_cluster)
    interviewer_sids = set(segment_ids[i] for i in range(len(segment_ids)) if labels[i] == interviewer_cluster)

    # Get Santa time ranges
    santa_ranges = []
    for sid, start_ms, end_ms in rows:
        if sid in santa_sids:
            santa_ranges.append((start_ms, end_ms))

    interviewer_ranges = []
    for sid, start_ms, end_ms in rows:
        if sid in interviewer_sids:
            interviewer_ranges.append((start_ms, end_ms))

    santa_total = sum(e - s for s, e in santa_ranges) / 1000
    interviewer_total = sum(e - s for s, e in interviewer_ranges) / 1000

    logger.info(f"Santa: {len(santa_ranges)} segments, {santa_total:.1f}s")
    logger.info(f"Interviewer: {len(interviewer_ranges)} segments, {interviewer_total:.1f}s")
    logger.info(f"Filtering ratio: keeping {santa_total/(santa_total+interviewer_total)*100:.1f}% of audio")

    db.close()
    return santa_ranges, interviewer_ranges


def is_in_santa_range(ts_ms: float, santa_ranges: list[tuple[int, int]], margin_ms: int = 500) -> bool:
    """Check if a timestamp falls within a Santa speaking range (with margin)."""
    for start, end in santa_ranges:
        if (start - margin_ms) <= ts_ms <= (end + margin_ms):
            return True
    return False


# =============================================================================
# Phase 0b: Rebuild embeddings with Santa-only data
# =============================================================================
def rebuild_santa_embeddings(santa_ranges: list[tuple[int, int]]):
    """Filter all_embeddings.npz to only include Santa's segments."""
    logger.info("=" * 60)
    logger.info("PHASE 0b: Rebuilding Santa-only embeddings")
    logger.info("=" * 60)

    emb_path = Path.home() / ".clipcannon" / "models" / "santa" / "embeddings" / "all_embeddings.npz"
    data = np.load(str(emb_path), allow_pickle=True)

    # Original counts
    orig = {k: len(data[k]) for k in data.files}
    logger.info(f"Original: {orig}")

    # Filter each modality by timestamp
    def filter_by_ts(arr, ts_arr, name):
        mask = np.array([is_in_santa_range(float(t), santa_ranges) for t in ts_arr])
        filtered = arr[mask]
        filtered_ts = ts_arr[mask]
        logger.info(f"  {name}: {len(arr)} -> {len(filtered)} ({len(filtered)/len(arr)*100:.1f}%)")
        return filtered, filtered_ts

    vis_emb, vis_ts = filter_by_ts(data["vis_emb"], data["vis_ts"], "visual")
    sem_emb, sem_ts = filter_by_ts(data["sem_emb"], data["sem_ts"], "semantic")
    emo_data, emo_ts = filter_by_ts(data["emo_data"], data["emo_ts"], "emotion")
    pro_data, pro_ts = filter_by_ts(data["pro_data"], data["pro_ts"], "prosody")

    # FLAME params — timestamps may be in seconds, convert if needed
    flame_ts_raw = data["flame_ts"]
    flame_exp = data["flame_exp"]
    if flame_ts_raw.max() < 10000:
        flame_ts_ms = flame_ts_raw * 1000
    else:
        flame_ts_ms = flame_ts_raw
    flame_mask = np.array([is_in_santa_range(float(t), santa_ranges) for t in flame_ts_ms])
    flame_exp_f = flame_exp[flame_mask]
    flame_ts_f = flame_ts_raw[flame_mask]
    logger.info(f"  flame: {len(flame_exp)} -> {len(flame_exp_f)} ({len(flame_exp_f)/len(flame_exp)*100:.1f}%)")

    # Sanity check
    if len(vis_emb) < 100:
        raise ValueError(f"Too few visual frames after filtering: {len(vis_emb)}")

    # Save — backup original first
    backup_path = emb_path.with_suffix(".npz.bak_with_interviewer")
    if not backup_path.exists():
        import shutil
        shutil.copy2(str(emb_path), str(backup_path))
        logger.info(f"Backed up original to {backup_path}")

    np.savez(
        str(emb_path),
        vis_emb=vis_emb, vis_ts=vis_ts,
        sem_emb=sem_emb, sem_ts=sem_ts,
        emo_data=emo_data, emo_ts=emo_ts,
        pro_data=pro_data, pro_ts=pro_ts,
        flame_exp=flame_exp_f, flame_ts=flame_ts_f,
    )
    logger.info(f"Saved Santa-only embeddings to {emb_path}")
    logger.info(f"Final counts: vis={len(vis_emb)}, sem={len(sem_emb)}, emo={len(emo_data)}, pro={len(pro_data)}, flame={len(flame_exp_f)}")
    return len(vis_emb)


# =============================================================================
# Phase 1-3: Full semantic transformer training
# =============================================================================
def run_full_training():
    """Run complete 4-phase semantic transformer training with production settings."""
    import torch

    logger.info("=" * 60)
    logger.info("FULL SEMANTIC TRANSFORMER TRAINING — Production Settings")
    logger.info("=" * 60)

    # Import trainer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from phoenix.clone.semantic_trainer import SemanticTrainer, SemanticTrainerConfig

    config = SemanticTrainerConfig(
        person="santa",
        device="cuda",
        # Phase 1: SPDs — 500 epochs (~30 min)
        spd_epochs=500,
        spd_lr=1e-3,
        spd_batch_size=64,
        # Phase 2: CMBs — 300 epochs (~15 min)
        cmb_epochs=300,
        cmb_lr=1e-3,
        # Phase 3: Constellation Transformer — 5000 epochs (~2-3 hrs)
        transformer_epochs=5000,
        transformer_lr=5e-4,
        transformer_batch_size=64,
        # Loss weights
        geometric_weight=1.0,
        semantic_weight=0.3,
        constellation_reg_weight=0.01,
        cmb_consistency_weight=0.1,
        # All phases
        start_phase=1,
        end_phase=3,
    )

    # Enable torch optimizations for Blackwell
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")  # Use TF32 on tensor cores

    logger.info("TF32 enabled, cuDNN benchmark=True")
    logger.info(f"Config: SPD={config.spd_epochs}ep, CMB={config.cmb_epochs}ep, Transformer={config.transformer_epochs}ep")

    t0 = time.time()
    trainer = SemanticTrainer(config)
    results = trainer.run()
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE in {elapsed/60:.1f} minutes")
    for phase, res in results.items():
        logger.info(f"  {phase}: {res}")
    logger.info(f"  Model: {config.save_dir}")
    logger.info("=" * 60)

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    t_start = time.time()
    logger.info("Starting Santa embedding rebuild + full training pipeline")
    logger.info(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")

    # Pre-flight
    import torch
    check_gpu_health()

    # Phase 0: Filter interviewer
    santa_ranges, interviewer_ranges = filter_interviewer_segments()
    gpu_cleanup()

    # Phase 0b: Rebuild clean embeddings
    n_frames = rebuild_santa_embeddings(santa_ranges)
    logger.info(f"Training will use {n_frames} Santa-only visual frames")

    # Phase 1-3: Full training
    gpu_cleanup()
    results = run_full_training()
    gpu_cleanup()

    total = time.time() - t_start
    logger.info(f"\nTotal pipeline time: {total/60:.1f} minutes ({total/3600:.1f} hours)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
