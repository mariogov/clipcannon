#!/usr/bin/env python3
"""LatentSync → EchoMimicV3 img2img refinement pipeline.

Two-stage pipeline that combines the best of both models:
  Stage 1: LatentSync produces phoneme-accurate lip-synced video (blurry)
  Stage 2: EchoMimicV3 + LoRA redraws the video at high quality,
           preserving LatentSync's lip positions via img2img denoising

The key insight: instead of starting EchoMimicV3 from pure noise, start
from LatentSync's VAE-encoded latents with partial noise added. The model
denoises LESS, preserving the structure (lip positions) while regenerating
sharp detail (face, beard, eyes, skin texture) from the LoRA identity.

The `strength` parameter (0.0-1.0) controls the balance:
  0.0 = pure LatentSync output (blurry but perfect lips)
  1.0 = pure EchoMimicV3 generation (sharp but generic lips)
  0.3-0.5 = sweet spot (preserves lip structure + sharp identity)

Usage:
    python scripts/lipsync_refine.py \
        --driver_video WEBCAM.mp4 \
        --audio VOCALS.wav \
        --reference_image SANTA_768.jpg \
        --output OUTPUT.mp4 \
        --strength 0.4 \
        --lora_path ~/.clipcannon/models/santa/echov3_lora_v2/final

Requirements:
    - LatentSync models at ~/.clipcannon/models/ (via ClipCannon)
    - EchoMimicV3 at ~/echomimic_v3/
    - v2 LoRA weights
"""

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/lipsync_refine.log")],
)
log = logging.getLogger("lipsync_refine")

# Paths
ECHOMIMIC_DIR = Path("/home/cabdru/echomimic_v3")
MODEL_PATH = ECHOMIMIC_DIR / "pretrained_weights" / "Wan2.1-Fun-V1.1-1.3B-InP"
TRANSFORMER_PATH = ECHOMIMIC_DIR / "pretrained_weights" / "echomimicv3-flash-pro" / "diffusion_pytorch_model.safetensors"
WAV2VEC_DIR = ECHOMIMIC_DIR / "pretrained_weights" / "chinese-wav2vec2-base"
CONFIG_PATH = ECHOMIMIC_DIR / "config" / "config.yaml"

sys.path.insert(0, str(ECHOMIMIC_DIR))


def stage1_latentsync(
    driver_video: Path, audio: Path, output: Path,
    steps: int = 25, guidance: float = 1.5, seed: int = 42,
) -> Path:
    """Stage 1: Run LatentSync for phoneme-accurate lip sync."""
    log.info("=" * 60)
    log.info("STAGE 1: LatentSync lip sync")
    log.info("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from clipcannon.avatar.lip_sync import LipSyncEngine

    engine = LipSyncEngine()
    result = engine.generate(
        video_path=driver_video,
        audio_path=audio,
        output_path=output,
        inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
    )
    log.info("LatentSync done: %s (%dms, %s)", result.video_path, result.duration_ms, result.resolution)

    # Free LatentSync GPU memory completely
    del engine
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    log.info("LatentSync VRAM freed: %.1fGB", torch.cuda.memory_allocated() / 1e9)

    return result.video_path


def load_video_frames(video_path: Path, max_frames: int = 249, size: int = 768):
    """Load video frames as (T, H, W, C) uint8 numpy array."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")

    return np.stack(frames)  # (T, H, W, C)


def stage2_echomimic_refine(
    latentsync_video: Path,
    audio: Path,
    reference_image: Path,
    output: Path,
    lora_path: Path,
    strength: float = 0.4,
    inference_steps: int = 30,
    guidance_scale: float = 3.5,
    resolution: int = 768,
    seed: int = 42,
):
    """Stage 2: EchoMimicV3 img2img refinement of LatentSync output.

    Encodes LatentSync video through VAE, adds partial noise controlled
    by `strength`, then denoises with EchoMimicV3 + LoRA to produce
    sharp identity while preserving LatentSync's lip positions.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    import librosa
    from PIL import Image
    import torchvision.transforms.functional as TF
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from einops import rearrange
    from peft import PeftModel

    log.info("=" * 60)
    log.info("STAGE 2: EchoMimicV3 img2img refinement (strength=%.2f)", strength)
    log.info("=" * 60)

    device = torch.device("cuda")
    config = OmegaConf.load(str(CONFIG_PATH))

    # Load LatentSync video frames — align to EchoMimicV3's temporal compression
    log.info("Loading LatentSync video frames...")
    ls_frames_raw = load_video_frames(latentsync_video, max_frames=249, size=resolution)
    # Align frame count: must satisfy (N-1) % 4 == 0 for VAE temporal compression
    raw_count = len(ls_frames_raw)
    video_length = ((raw_count - 1) // 4) * 4 + 1  # nearest valid: 49, 97, 145, ...
    video_length = min(video_length, raw_count)
    ls_frames = ls_frames_raw[:video_length]
    latent_t = (video_length - 1) // 4 + 1
    log.info("Loaded %d frames at %dx%d", video_length, resolution, resolution)

    # ---- VAE encode LatentSync output (FP32, isolated) ----
    from src.wan_vae import AutoencoderKLWan
    log.info("VAE encoding LatentSync frames (FP32)...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(str(MODEL_PATH), config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(torch.float32).to(device)

    # Convert frames to tensor: (1, C, T, H, W) normalized to [-1, 1]
    video_tensor = torch.from_numpy(ls_frames).permute(3, 0, 1, 2).float()  # (C, T, H, W)
    video_tensor = video_tensor / 127.5 - 1.0
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)

    with torch.no_grad():
        init_latents = vae.encode(video_tensor).latent_dist.sample()
    log.info("Init latents: %s", init_latents.shape)

    # Also encode reference image for CLIP and inpainting condition
    ref_img = Image.open(reference_image).convert("RGB").resize((resolution, resolution))
    ref_tensor = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1).float() / 127.5 - 1.0
    ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(2).to(device)  # (1, C, 1, H, W)
    with torch.no_grad():
        ref_latent = vae.encode(ref_tensor).latent_dist.sample()

    del vae; gc.collect(); torch.cuda.empty_cache()
    log.info("VAE done. VRAM: %.1fGB", torch.cuda.memory_allocated() / 1e9)

    # ---- T5 text encoding ----
    from src.wan_text_encoder import WanT5EncoderModel
    from transformers import AutoTokenizer
    log.info("Loading T5...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(str(MODEL_PATH), config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(str(MODEL_PATH), config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    prompt = (
        "SANTA An older man dressed as Santa Claus with a white beard, "
        "round glasses, and red suit, speaking naturally, mouth moving "
        "with words, natural facial expressions, warm interview lighting."
    )
    text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    seq_lens = text_inputs.attention_mask.sum(dim=1).tolist()
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device), attention_mask=text_inputs.attention_mask.to(device))[0]
    text_context = [prompt_embeds[0, :seq_lens[0]]]
    del text_encoder, tokenizer; gc.collect(); torch.cuda.empty_cache()
    log.info("T5 done")

    # ---- CLIP image encoding ----
    from src.wan_image_encoder import CLIPModel
    log.info("Loading CLIP...")
    clip_encoder = CLIPModel.from_pretrained(
        os.path.join(str(MODEL_PATH), config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(torch.bfloat16).to(device).eval()
    clip_image = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        clip_fea = clip_encoder([clip_image[:, None, :, :]])
    del clip_encoder; gc.collect(); torch.cuda.empty_cache()
    log.info("CLIP done: %s", clip_fea.shape)

    # ---- Audio encoding ----
    from src.wav2vec2 import Wav2Vec2Model
    from transformers import Wav2Vec2FeatureExtractor

    def loudness_norm(audio, sr, target=-23.0):
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if loudness == float('-inf'):
            return audio
        return pyln.normalize.loudness(audio, loudness, target)

    log.info("Loading Wav2Vec...")
    audio_encoder = Wav2Vec2Model.from_pretrained(str(WAV2VEC_DIR), local_files_only=True).to(device).eval()
    wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(str(WAV2VEC_DIR), local_files_only=True)

    mel_input, sr = librosa.load(str(audio), sr=16000)
    mel_input = loudness_norm(mel_input, sr)
    mel_input = mel_input[:int(video_length / 25 * sr)]
    audio_feature = np.squeeze(wav2vec_fe(mel_input, sampling_rate=16000).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=video_length, output_hidden_states=True)
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

    indices = (torch.arange(2 * 2 + 1) - 2) * 1
    center_indices = torch.arange(0, video_length, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0] - 1)
    audio_embeds = audio_emb[center_indices].unsqueeze(0).to(device, dtype=torch.bfloat16)

    del audio_encoder, wav2vec_fe; gc.collect(); torch.cuda.empty_cache()
    log.info("Audio done: %s", audio_embeds.shape)

    # ---- Load transformer + LoRA ----
    from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
    log.info("Loading transformer...")
    transformer = WanTransformer.from_pretrained(
        os.path.join(str(MODEL_PATH), config['transformer_additional_kwargs'].get('transformer_subpath', './')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=torch.bfloat16,
    )
    transformer.load_state_dict(load_file(str(TRANSFORMER_PATH)), strict=False)
    transformer = transformer.to(device)

    if lora_path and lora_path.exists():
        log.info("Loading LoRA from %s", lora_path)
        transformer = PeftModel.from_pretrained(transformer, str(lora_path))
        transformer = transformer.merge_and_unload()
        log.info("LoRA merged")

    transformer.eval()
    log.info("Transformer loaded: %.1fGB", torch.cuda.memory_allocated() / 1e9)

    # ---- Scheduler ----
    from src.fm_solvers import FlowDPMSolverMultistepScheduler
    from src.utils import filter_kwargs
    scheduler_config = OmegaConf.to_container(config['scheduler_kwargs'])
    scheduler_config['shift'] = 1
    scheduler = FlowDPMSolverMultistepScheduler(
        **filter_kwargs(FlowDPMSolverMultistepScheduler, scheduler_config)
    )

    # ---- img2img denoising loop ----
    log.info("Starting img2img denoising: %d steps, strength=%.2f", inference_steps, strength)

    scheduler.set_timesteps(inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Determine start step based on strength
    # strength=0.0 → start at last step (no denoising, pure LatentSync)
    # strength=1.0 → start at first step (full denoising, pure EchoMimicV3)
    start_step = max(0, int(inference_steps * (1.0 - strength)))
    timesteps = timesteps[start_step:]
    log.info("Denoising from step %d/%d (%d steps total)", start_step, inference_steps, len(timesteps))

    # Add noise to init_latents at the starting timestep
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)

    init_latents_bf16 = init_latents.to(torch.bfloat16)
    noise_bf16 = noise.to(torch.bfloat16)

    if len(timesteps) > 0:
        t_start = timesteps[0]
        sigma = t_start / 1000.0
        latents = (1 - sigma) * init_latents_bf16 + sigma * noise_bf16
    else:
        latents = init_latents_bf16

    # Prepare inpainting condition y
    B, C, T, H, W = init_latents.shape
    mask_cond = torch.ones(B, 4, T, H, W, device=device, dtype=torch.bfloat16)
    mask_cond[:, :, 0, :, :] = 0  # frame 0 visible
    masked_lat = torch.zeros(B, C, T, H, W, device=device, dtype=torch.bfloat16)
    masked_lat[:, :, 0:1, :, :] = ref_latent.to(torch.bfloat16)
    y = torch.cat([mask_cond, masked_lat], dim=1)

    context = (text_context, audio_embeds, T, None)
    seq_len = T * H * W // 4

    # Denoising loop
    for i, t in enumerate(timesteps):
        timestep = t.unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            noise_pred = transformer(
                x=latents,
                t=timestep.to(torch.int64),
                context=context,
                seq_len=seq_len,
                clip_fea=clip_fea.to(device, dtype=torch.bfloat16),
                y=y,
            )

        if isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 5 == 0:
            log.info("  Step %d/%d", i + 1, len(timesteps))

    del transformer; gc.collect(); torch.cuda.empty_cache()
    log.info("Denoising done. Decoding...")

    # ---- VAE decode ----
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(str(MODEL_PATH), config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(torch.float32).to(device)

    with torch.no_grad():
        frames = vae.decode(latents.to(torch.float32)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float().numpy()

    del vae; gc.collect(); torch.cuda.empty_cache()

    # frames shape: (1, C, T, H, W) → save as video
    frames = frames[0]  # (C, T, H, W)
    frames = (frames * 255).astype(np.uint8)
    frames = frames.transpose(1, 2, 3, 0)  # (T, H, W, C)

    log.info("Decoded %d frames at %dx%d", frames.shape[0], frames.shape[2], frames.shape[1])

    # Write frames to video with FFmpeg
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix="refine_"))
    try:
        import cv2
        for i, frame in enumerate(frames):
            cv2.imwrite(str(frame_dir / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "25",
            "-i", str(frame_dir / "%05d.png"),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            str(output),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        log.info("Output saved: %s (%.1f MB)", output, output.stat().st_size / 1e6)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    return output


def main():
    parser = argparse.ArgumentParser(description="LatentSync → EchoMimicV3 refinement")
    parser.add_argument("--driver_video", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--reference_image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=str(Path.home() / ".clipcannon/models/santa/echov3_lora_v2/final"))
    parser.add_argument("--strength", type=float, default=0.4,
                        help="0.0=pure LatentSync, 1.0=pure EchoMimicV3, 0.3-0.5=sweet spot")
    parser.add_argument("--latentsync_steps", type=int, default=25)
    parser.add_argument("--refine_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_latentsync", action="store_true",
                        help="Skip stage 1 if LatentSync output already exists at --driver_video")
    args = parser.parse_args()

    driver = Path(args.driver_video)
    audio = Path(args.audio)
    ref = Path(args.reference_image)
    out = Path(args.output)
    lora = Path(args.lora_path)

    t0 = time.time()

    if args.skip_latentsync:
        ls_output = driver
        log.info("Skipping LatentSync (using existing: %s)", ls_output)
    else:
        ls_output = out.parent / f"{out.stem}_latentsync.mp4"
        stage1_latentsync(driver, audio, ls_output, steps=args.latentsync_steps, seed=args.seed)

    stage2_echomimic_refine(
        latentsync_video=ls_output,
        audio=audio,
        reference_image=ref,
        output=out,
        lora_path=lora,
        strength=args.strength,
        inference_steps=args.refine_steps,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        seed=args.seed,
    )

    log.info("Pipeline complete in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
