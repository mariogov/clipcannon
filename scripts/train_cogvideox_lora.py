#!/usr/bin/env python3
"""CogVideoX-2b LoRA training on pre-encoded Santa video latents.

Videos are pre-encoded by the VAE (separately, to avoid WSL2 CUDA driver crash
when VAE and transformer share GPU). This script loads the latents and trains
the transformer LoRA weights only.

Usage:
    python scripts/train_cogvideox_lora.py --steps 1000

VRAM: ~12GB (transformer BF16 + LoRA + optimizer + activations)
"""
import gc
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/lora_train.log")])
log = logging.getLogger("lora")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from diffusers import CogVideoXTransformer3DModel, CogVideoXDPMScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from peft import LoraConfig, get_peft_model
from transformers import T5EncoderModel, T5Tokenizer


class LatentDataset(Dataset):
    """Loads pre-encoded video latents from disk."""
    def __init__(self, latent_dir: str):
        self.files = sorted(Path(latent_dir).glob("*.pt"))
        if not self.files:
            raise ValueError(f"No latent files in {latent_dir}")
        log.info("Dataset: %d pre-encoded latents", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=True)
        # Sample from the latent distribution
        mean, logvar = d["mean"].squeeze(0), d["logvar"].squeeze(0)
        std = torch.exp(0.5 * logvar)
        latent = mean + std * torch.randn_like(std)
        return latent  # (16, T, H, W)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/cabdru/.cache/huggingface/hub/cogvideox-2b")
    p.add_argument("--latents", default="/home/cabdru/.clipcannon/models/santa/cogvideo_training/latents")
    p.add_argument("--output", default="/home/cabdru/.clipcannon/models/santa/lora")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-every", type=int, default=250)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    # --- Load T5 for prompt encoding, then free it ---
    log.info("Encoding prompt with T5...")
    tokenizer = T5Tokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)

    prompt = ("SANTA A close-up video of Santa Claus with a white beard, round glasses, "
              "and red suit talking emotionally in an interview setting with warm lighting")
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length",
                       max_length=tokenizer.model_max_length, truncation=True)
    with torch.no_grad():
        prompt_embeds = text_encoder(inputs.input_ids.to(device))[0]  # (1, seq_len, hidden)

    # Free T5 — ~5GB
    del text_encoder, tokenizer, inputs
    gc.collect(); torch.cuda.empty_cache()
    log.info("T5 freed. VRAM: %.1fGB", torch.cuda.memory_allocated() / 1e9)

    # --- Load transformer + apply LoRA ---
    log.info("Loading transformer...")
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.enable_gradient_checkpointing()

    # Maximum identity lock — LoRA adapts ALL linear layers, not just attention.
    # alpha = 2× rank makes LoRA dominant over base weights.
    # No dropout — we WANT overfitting to this specific Santa.
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "proj_in", "proj_out",
                        "ff.net.0.proj", "ff.net.2"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    log.info("LoRA: rank=%d, trainable=%d (%.2f%% of %d)", args.rank, trainable,
             100 * trainable / total, total)
    log.info("VRAM after transformer: %.1fGB", torch.cuda.memory_allocated() / 1e9)

    # --- Scheduler ---
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    vae_scaling = 1.15258426  # from VAE config

    # --- Dataset ---
    dataset = LatentDataset(args.latents)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2,
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # --- Training ---
    log.info("Starting training: %d steps, lr=%.1e, rank=%d", args.steps, args.lr, args.rank)
    transformer.train()
    step = 0
    best_loss = float("inf")
    loss_accum = 0.0
    t0 = time.time()

    while step < args.steps:
        for latents in loader:
            if step >= args.steps:
                break

            # latents: (B, 16, T, H, W) — already scaled by VAE
            latents = latents.to(device, dtype=torch.bfloat16) * vae_scaling

            # Reshape for transformer: (B, T, C, H, W)
            latents = latents.permute(0, 2, 1, 3, 4)

            noise = torch.randn_like(latents)
            B = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Expand prompt embeds to batch
            encoder_hidden_states = prompt_embeds.expand(B, -1, -1)

            # Forward
            model_output = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                return_dict=False,
            )[0]

            # Velocity prediction loss
            model_pred = scheduler.get_velocity(model_output, noisy_latents, timesteps)
            target = latents

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            step += 1

            if step % 25 == 0:
                avg = loss_accum / 25 * args.grad_accum
                elapsed = time.time() - t0
                vram = torch.cuda.memory_allocated() / 1e9
                peak = torch.cuda.max_memory_allocated() / 1e9
                log.info("Step %d/%d | loss=%.4f | lr=%.2e | %.0fs | VRAM=%.1f/%.1fGB",
                         step, args.steps, avg, lr_sched.get_last_lr()[0], elapsed, vram, peak)
                if avg < best_loss:
                    best_loss = avg
                loss_accum = 0.0

            if step % args.save_every == 0:
                save_path = os.path.join(args.output, f"checkpoint-{step}")
                transformer.save_pretrained(save_path)
                log.info("Checkpoint: %s", save_path)

    # Final save
    final = os.path.join(args.output, "final")
    transformer.save_pretrained(final)
    elapsed = time.time() - t0
    log.info("Done: %d steps in %.1f min | best_loss=%.4f | saved: %s", args.steps, elapsed/60, best_loss, final)


if __name__ == "__main__":
    main()
