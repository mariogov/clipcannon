#!/usr/bin/env python3
"""MuseTalk 1.5 post-processing for EchoMimicV3 output.

After EchoMimicV3 generates a video with face, body, and expressions,
MuseTalk 1.5 fixes ONLY the mouth region with phoneme-accurate lip sync.

MuseTalk 1.5 is NOT diffusion — it's single-step latent inpainting,
so there's no blur or quality degradation. It only touches the mouth.

Pipeline: EchoMimicV3 output → MuseTalk 1.5 → final video with precise lip sync

Usage:
    python scripts/musetalk_postprocess.py \
        --video /path/to/echomimic_output.mp4 \
        --audio /path/to/audio.wav \
        --output /path/to/final_output.mp4

    # Or with explicit model paths:
    python scripts/musetalk_postprocess.py \
        --video output.mp4 --audio audio.wav --output final.mp4 \
        --musetalk-dir /home/cabdru/MuseTalk
"""

import argparse
import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("musetalk_post")

MUSETALK_DIR = Path("/home/cabdru/MuseTalk")
MUSETALK_MODELS = MUSETALK_DIR / "models"


def validate_paths(video_path: Path, audio_path: Path, musetalk_dir: Path) -> bool:
    """Validate all required files exist."""
    errors = []
    if not video_path.exists():
        errors.append(f"Video not found: {video_path}")
    if not audio_path.exists():
        errors.append(f"Audio not found: {audio_path}")
    if not musetalk_dir.exists():
        errors.append(f"MuseTalk directory not found: {musetalk_dir}")

    unet_path = musetalk_dir / "models" / "musetalkV15" / "unet.pth"
    if not unet_path.exists():
        errors.append(f"MuseTalk 1.5 UNet not found: {unet_path}")

    vae_path = musetalk_dir / "models" / "sd-vae" / "diffusion_pytorch_model.bin"
    if not vae_path.exists():
        errors.append(f"SD VAE not found: {vae_path}")

    whisper_path = musetalk_dir / "models" / "whisper"
    if not whisper_path.exists():
        errors.append(f"Whisper model not found: {whisper_path}")

    for err in errors:
        log.error(err)
    return len(errors) == 0


def create_inference_config(
    video_path: Path, audio_path: Path, config_dir: Path
) -> Path:
    """Create a MuseTalk inference YAML config for this specific task."""
    config_content = f"""task_0:
  video_path: "{video_path}"
  audio_path: "{audio_path}"
"""
    config_path = config_dir / "musetalk_task.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def run_musetalk(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    musetalk_dir: Path,
    gpu_id: int = 0,
    batch_size: int = 8,
    use_float16: bool = True,
    fps: int = 25,
) -> bool:
    """Run MuseTalk 1.5 inference as a subprocess.

    Uses subprocess to isolate MuseTalk's CUDA context from EchoMimicV3,
    preventing WSL2 driver crashes from concurrent GPU contexts.
    """
    with tempfile.TemporaryDirectory(prefix="musetalk_") as tmpdir:
        tmpdir = Path(tmpdir)
        result_dir = tmpdir / "results"
        result_dir.mkdir()

        # Create inference config
        config_path = create_inference_config(video_path, audio_path, tmpdir)

        # Build MuseTalk command
        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--inference_config", str(config_path),
            "--result_dir", str(result_dir),
            "--unet_model_path", str(musetalk_dir / "models" / "musetalkV15" / "unet.pth"),
            "--unet_config", str(musetalk_dir / "models" / "musetalkV15" / "musetalk.json"),
            "--version", "v15",
            "--gpu_id", str(gpu_id),
            "--batch_size", str(batch_size),
            "--fps", str(fps),
        ]
        if use_float16:
            cmd.append("--use_float16")

        log.info("Running MuseTalk 1.5...")
        log.info("  Video: %s", video_path)
        log.info("  Audio: %s", audio_path)
        log.info("  Config: %s", config_path)

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = ""

        result = subprocess.run(
            cmd,
            cwd=str(musetalk_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            log.error("MuseTalk failed (exit %d):", result.returncode)
            log.error("  stdout: %s", result.stdout[-2000:] if result.stdout else "(empty)")
            log.error("  stderr: %s", result.stderr[-2000:] if result.stderr else "(empty)")
            return False

        # Find the output video
        output_videos = list(result_dir.rglob("*.mp4"))
        # Prefer non-concat version
        output_candidates = [v for v in output_videos if "_concat" not in v.name]
        if not output_candidates:
            output_candidates = output_videos

        if not output_candidates:
            log.error("MuseTalk produced no output video in %s", result_dir)
            log.error("  Files found: %s", list(result_dir.rglob("*")))
            return False

        # Copy best output to target path
        best_output = output_candidates[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_output), str(output_path))
        log.info("MuseTalk output saved to %s", output_path)
        return True


def postprocess_video(
    echomimic_video: Path,
    audio_path: Path,
    output_path: Path,
    musetalk_dir: Path = MUSETALK_DIR,
    gpu_id: int = 0,
    fps: int = 25,
) -> bool:
    """Full post-processing pipeline: EchoMimicV3 output → MuseTalk 1.5 → final.

    1. Validates inputs
    2. Runs MuseTalk 1.5 on the EchoMimicV3 video + original audio
    3. The result has precise phoneme-accurate lip sync from MuseTalk
       while keeping EchoMimicV3's face/body/expression quality
    """
    log.info("=" * 60)
    log.info("MuseTalk 1.5 Post-Processing")
    log.info("=" * 60)

    if not validate_paths(echomimic_video, audio_path, musetalk_dir):
        return False

    success = run_musetalk(
        video_path=echomimic_video,
        audio_path=audio_path,
        output_path=output_path,
        musetalk_dir=musetalk_dir,
        gpu_id=gpu_id,
        fps=fps,
    )

    if success:
        # Verify output
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            log.info("Post-processing complete: %s (%.1f MB)", output_path, size_mb)
        else:
            log.error("Output file not found after processing")
            return False

    return success


def main():
    parser = argparse.ArgumentParser(
        description="MuseTalk 1.5 post-processing for EchoMimicV3 output"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to EchoMimicV3 output video")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file for lip sync")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for final output video")
    parser.add_argument("--musetalk-dir", type=str, default=str(MUSETALK_DIR),
                        help="Path to MuseTalk installation")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output video FPS")
    args = parser.parse_args()

    success = postprocess_video(
        echomimic_video=Path(args.video),
        audio_path=Path(args.audio),
        output_path=Path(args.output),
        musetalk_dir=Path(args.musetalk_dir),
        gpu_id=args.gpu_id,
        fps=args.fps,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
