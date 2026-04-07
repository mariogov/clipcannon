#!/usr/bin/env python3
"""EchoMimicV3 LoRA training with EXACT tensor shapes from the inference pipeline.

Shapes captured from pipeline forward:
  x:          (1, 16, 13, 60, 60)   - latent
  t:          (1,) int64             - timestep
  context[0]: list                   - text embeddings from T5
  context[1]: (1, 49, 5, 12, 768)   - audio embeds (B, F, window, layers, dim)
  context[2]: int                    - latent temporal frames (13)
  context[3]: None                   - IP mask
  seq_len:    11700                  - 13*60*60/4
  clip_fea:   (1, 257, 1280)        - CLIP image features
  y:          (1, 20, 13, 60, 60)   - mask(4ch) + masked_latent(16ch)
"""
import gc, logging, os, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/echov3_lora_train.log")])
log = logging.getLogger("echov3")

sys.path.insert(0, "/home/cabdru/echomimic_v3")

import numpy as np, librosa, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from peft import LoraConfig, get_peft_model

MODEL_PATH = "/home/cabdru/echomimic_v3/pretrained_weights/Wan2.1-Fun-V1.1-1.3B-InP"
TRANSFORMER_PATH = "/home/cabdru/echomimic_v3/pretrained_weights/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"
WAV2VEC_DIR = "/home/cabdru/echomimic_v3/pretrained_weights/chinese-wav2vec2-base"
DATA_DIR = "/home/cabdru/echomimic_v3/datasets/santa_training"
COND_DIR = "/home/cabdru/echomimic_v3/datasets/santa_training/conditioning_v2"
OUTPUT_DIR = "/home/cabdru/.clipcannon/models/santa/echov3_lora"
CONFIG_PATH = "/home/cabdru/echomimic_v3/config/config.yaml"


def loudness_norm(audio, sr, target=-23.0):
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if loudness == float('-inf'): return audio
    return pyln.normalize.loudness(audio, loudness, target)


def pre_encode_all():
    """Pre-encode ALL conditioning with exact shapes matching the pipeline."""
    os.makedirs(COND_DIR, exist_ok=True)
    config = OmegaConf.load(CONFIG_PATH)
    device = torch.device("cuda")
    data_dir = Path(DATA_DIR)
    samples = sorted([p.stem for p in (data_dir / "imgs").glob("*.jpg")])

    prompt = "An older man dressed as Santa Claus with a white beard, round glasses, and red suit is talking emotionally."
    video_length = 49  # frames per clip at 25fps
    latent_t = (video_length - 1) // 4 + 1  # = 13 (temporal compression ratio 4)

    # ---- T5 text encoding (encode once, reuse) ----
    from src.wan_text_encoder import WanT5EncoderModel
    from transformers import AutoTokenizer
    log.info("Loading T5...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(MODEL_PATH, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    seq_lens = text_inputs.attention_mask.sum(dim=1).tolist()
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device), attention_mask=text_inputs.attention_mask.to(device))[0]
    # Pipeline returns list of 2D tensors: [u[:seq_len] for each batch item]
    text_context_cpu = [prompt_embeds[0, :seq_lens[0]].cpu()]  # list of (seq_len, dim)
    del text_encoder, tokenizer; gc.collect(); torch.cuda.empty_cache()
    log.info("T5 done. VRAM: %.1fGB", torch.cuda.memory_allocated()/1e9)

    # ---- CLIP image encoding ----
    import torchvision.transforms.functional as TF
    from src.wan_image_encoder import CLIPModel
    log.info("Loading CLIP...")
    clip_encoder = CLIPModel.from_pretrained(
        os.path.join(MODEL_PATH, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(torch.bfloat16).to(device).eval()

    clip_features = {}
    for idx, name in enumerate(samples):
        img = Image.open(data_dir / "imgs" / f"{name}.jpg").convert("RGB").resize((480, 480))
        clip_image = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            clip_features[name] = clip_encoder([clip_image[:, None, :, :]]).cpu()
        if (idx+1) % 50 == 0: log.info("  CLIP: %d/%d", idx+1, len(samples))
    del clip_encoder; gc.collect(); torch.cuda.empty_cache()
    log.info("CLIP done: %s per sample", clip_features[samples[0]].shape)

    # ---- Audio encoding with windowing ----
    from src.wav2vec2 import Wav2Vec2Model
    from transformers import Wav2Vec2FeatureExtractor
    log.info("Loading Wav2Vec...")
    audio_encoder = Wav2Vec2Model.from_pretrained(WAV2VEC_DIR, local_files_only=True).to(device).eval()
    wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_DIR, local_files_only=True)

    audio_features = {}
    for idx, name in enumerate(samples):
        mel_input, sr = librosa.load(str(data_dir / "audios" / f"{name}.wav"), sr=16000)
        mel_input = loudness_norm(mel_input, sr)
        mel_input = mel_input[:int(video_length / 25 * sr)]
        audio_feature = np.squeeze(wav2vec_fe(mel_input, sampling_rate=16000).input_values)
        audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings = audio_encoder(audio_feature, seq_len=video_length, output_hidden_states=True)
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        # Window the audio (from infer_flash.py lines 363-370)
        indices = (torch.arange(2*2+1) - 2) * 1
        center_indices = torch.arange(0, video_length, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0]-1)
        audio_windowed = audio_emb[center_indices].unsqueeze(0)  # (1, F, 5, 12, 768)
        audio_features[name] = audio_windowed.cpu()
        if (idx+1) % 50 == 0: log.info("  Audio: %d/%d", idx+1, len(samples))
    del audio_encoder, wav2vec_fe; gc.collect(); torch.cuda.empty_cache()
    log.info("Audio done: %s per sample", audio_features[samples[0]].shape)

    # ---- VAE latent encoding (FP32, isolated) ----
    from src.wan_vae import AutoencoderKLWan
    log.info("Loading VAE (FP32)...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(MODEL_PATH, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(torch.float32).to(device)

    latents = {}
    for idx, name in enumerate(samples):
        img = Image.open(data_dir / "imgs" / f"{name}.jpg").convert("RGB").resize((480, 480))
        # Create single-frame video tensor (B, C, T=1, H, W)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        tensor = tensor.unsqueeze(0).unsqueeze(2).to(device)  # (1, 3, 1, 480, 480)
        with torch.no_grad():
            lat = vae.encode(tensor).latent_dist.sample()
        latents[name] = lat.cpu()
        if (idx+1) % 50 == 0: log.info("  VAE: %d/%d", idx+1, len(samples))
    del vae; gc.collect(); torch.cuda.empty_cache()
    log.info("VAE done: %s per sample", latents[samples[0]].shape)

    # ---- Save everything ----
    for name in samples:
        torch.save({
            "text_context": text_context_cpu,
            "clip_fea": clip_features[name],      # (1, 257, 1280)
            "audio_emb": audio_features[name],     # (1, 49, 5, 12, 768)
            "latent": latents[name],               # (1, 16, 1, 60, 60)
            "latent_t": latent_t,                  # 13
            "video_length": video_length,           # 49
        }, Path(COND_DIR) / f"{name}.pt")
    log.info("All conditioning saved: %d samples in %s", len(samples), COND_DIR)


class TrainDataset(Dataset):
    def __init__(self, cond_dir):
        self.files = sorted(Path(cond_dir).glob("*.pt"))
        if not self.files: raise ValueError(f"No data in {cond_dir}")
        log.info("Dataset: %d samples", len(self.files))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)


def train_lora(steps=2000, rank=128, lr=2e-4, save_every=500):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")
    config = OmegaConf.load(CONFIG_PATH)

    # Load transformer
    log.info("Loading transformer...")
    from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
    transformer = WanTransformer.from_pretrained(
        os.path.join(MODEL_PATH, config['transformer_additional_kwargs'].get('transformer_subpath', './')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=torch.bfloat16,
    )
    transformer.load_state_dict(load_file(TRANSFORMER_PATH), strict=False)
    transformer = transformer.to(device)
    transformer.gradient_checkpointing = True
    log.info("Transformer: %.1fGB", torch.cuda.memory_allocated()/1e9)

    # Apply LoRA
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank*2,
        target_modules=["q", "k", "v", "o", "q_audio", "k_audio", "v_audio", "k_img", "v_img"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    log.info("LoRA rank=%d: %d trainable (%.2f%%)", rank, trainable, 100*trainable/sum(p.numel() for p in transformer.parameters()))

    dataset = TrainDataset(COND_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])  # Don't batch — each sample is pre-batched

    optimizer = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    log.info("Training: %d steps, rank=%d, lr=%.1e", steps, rank, lr)
    transformer.train()
    step, best_loss, loss_accum, t0 = 0, float("inf"), 0.0, time.time()

    while step < steps:
        for sample in loader:
            if step >= steps: break

            latent = sample["latent"].to(device, dtype=torch.bfloat16)  # (1, 16, 1, 60, 60)
            clip_fea = sample["clip_fea"].to(device, dtype=torch.bfloat16)  # (1, 257, 1280)
            audio_emb = sample["audio_emb"].to(device, dtype=torch.bfloat16)  # (1, 49, 5, 12, 768)
            latent_t = sample["latent_t"]  # 13

            # Expand latent to match expected temporal dim (repeat frame to fill T=13)
            latent_expanded = latent.expand(-1, -1, latent_t, -1, -1)  # (1, 16, 13, 60, 60)
            B, C, T, H, W = latent_expanded.shape

            # Flow matching noise
            noise = torch.randn_like(latent_expanded)
            t_idx = torch.randint(0, 1000, (1,)).item()
            sigma = t_idx / 1000.0
            noisy = (1 - sigma) * latent_expanded + sigma * noise

            # Inpainting condition y (mask + masked_latent)
            mask_cond = torch.ones(B, 4, T, H, W, device=device, dtype=torch.bfloat16)
            masked_lat = torch.zeros(B, C, T, H, W, device=device, dtype=torch.bfloat16)
            y = torch.cat([mask_cond, masked_lat], dim=1)  # (1, 20, 13, 60, 60)

            # Context tuple: (text_context_list, audio_emb, latent_t, ip_mask)
            # text_context is a list of 2D tensors [(seq_len, dim)]
            text_ctx = sample["text_context"]
            text_ctx = [t.to(device, dtype=torch.bfloat16) for t in text_ctx]
            context_tuple = (text_ctx, audio_emb, latent_t, None)

            seq_len = T * H * W // 4  # 11700

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = transformer(
                    x=noisy,
                    t=torch.tensor([t_idx], device=device, dtype=torch.int64),
                    context=context_tuple,
                    seq_len=seq_len,
                    clip_fea=clip_fea,
                    y=y,
                )
            if isinstance(output, tuple):
                output = output[0]

            target = latent_expanded - noise
            loss = F.mse_loss(output.float(), target.float()) / 2
            loss.backward()
            loss_accum += loss.item()

            if (step+1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_([p for p in transformer.parameters() if p.requires_grad], 1.0)
                optimizer.step(); lr_sched.step(); optimizer.zero_grad()

            step += 1
            if step % 25 == 0:
                avg = loss_accum / 25 * 2
                if avg < best_loss: best_loss = avg
                log.info("Step %d/%d | loss=%.4f | lr=%.2e | %.0fs | VRAM=%.1f/%.1fGB",
                         step, steps, avg, lr_sched.get_last_lr()[0], time.time()-t0,
                         torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
                loss_accum = 0.0
            if step % save_every == 0:
                transformer.save_pretrained(f"{OUTPUT_DIR}/checkpoint-{step}")
                log.info("Checkpoint saved")

    transformer.save_pretrained(f"{OUTPUT_DIR}/final")
    log.info("Done: %d steps in %.1f min | best=%.4f", steps, (time.time()-t0)/60, best_loss)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["encode", "train", "all"], default="all")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    if args.phase in ("encode", "all"):
        pre_encode_all()
    if args.phase in ("train", "all"):
        train_lora(steps=args.steps, rank=args.rank, lr=args.lr)
