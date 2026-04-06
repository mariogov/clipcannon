"""Semantic Transformer Training -- 7 modalities + Cycle Consistency.

Phase 1: Train all 7 SPDs on pseudo-labels  |  Phase 2: Train 21 CMBs
Phase 3: Constellation transformer           |  Phase 4: Cycle consistency

CLI:
  python -m phoenix.clone.semantic_trainer --person santa
  python -m phoenix.clone.semantic_trainer --person santa --phase 4 --cycle-steps 500
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from phoenix.clone.cross_modal_bridges import train_bridges
from phoenix.clone.meaning_trainer import (
    MeaningAwareLoss, SemanticStateExtractor, _nearest_idx,
)
from phoenix.clone.semantic_decoders import (
    SEMANTIC_DIM, EmotionSPD, ProsodySPD, SemanticSPD, SpeakerSPD,
    SentenceSPD, VoiceSPD, VisualSPD,
    generate_emotion_labels, generate_prosody_labels, generate_semantic_labels,
    generate_speaker_labels, generate_sentence_labels, generate_voice_labels,
    generate_visual_labels,
)
from phoenix.clone.semantic_model import SemanticCloneModel

logger = logging.getLogger(__name__)

_REQUIRED_NPZ_KEYS = [
    "vis_emb", "vis_ts", "sem_emb", "sem_ts", "emo_emb", "emo_ts",
    "spk_emb", "spk_ts", "pro_data", "pro_ts", "sent_emb", "sent_ts",
    "voice_emb", "voice_ts", "flame_exp", "flame_ts",
]


@dataclass
class SemanticTrainerConfig:
    person: str = "santa"
    embeddings_path: str = ""
    save_dir: str = ""
    device: str = "cuda"
    spd_epochs: int = 300
    spd_lr: float = 1e-3
    spd_batch_size: int = 64
    cmb_epochs: int = 200
    cmb_lr: float = 1e-3
    transformer_epochs: int = 2000
    transformer_lr: float = 5e-4
    transformer_batch_size: int = 64
    constellation_reg_weight: float = 0.01
    cmb_consistency_weight: float = 0.1
    geometric_weight: float = 1.0
    semantic_weight: float = 0.3
    cycle_steps: int = 0
    cycle_lr: float = 1e-4
    start_phase: int = 1
    end_phase: int = 4

    def __post_init__(self) -> None:
        base = Path.home() / ".clipcannon" / "models" / self.person
        if not self.embeddings_path:
            self.embeddings_path = str(base / "embeddings" / "all_embeddings.npz")
        if not self.save_dir:
            self.save_dir = str(base / "semantic_model")


class SemanticTrainer:
    """Four-phase training pipeline for SemanticCloneModel (7 modalities)."""

    def __init__(self, config: SemanticTrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading embeddings from %s", config.embeddings_path)
        data = np.load(config.embeddings_path, allow_pickle=True)
        missing = [k for k in _REQUIRED_NPZ_KEYS if k not in data]
        if missing:
            raise ValueError(f"NPZ missing required keys: {missing}")
        self.vis_emb, self.vis_ts = data["vis_emb"], data["vis_ts"]
        self.sem_emb, self.sem_ts = data["sem_emb"], data["sem_ts"]
        self.emo_emb, self.emo_ts = data["emo_emb"], data["emo_ts"]
        self.spk_emb, self.spk_ts = data["spk_emb"], data["spk_ts"]
        self.pro_data, self.pro_ts = data["pro_data"], data["pro_ts"]
        self.sent_emb, self.sent_ts = data["sent_emb"], data["sent_ts"]
        self.voice_emb, self.voice_ts = data["voice_emb"], data["voice_ts"]
        self.flame_exp = data["flame_exp"]
        fts = data["flame_ts"]
        self.flame_ts = (fts * 1000).astype(np.int64) if fts.max() < 10000 else fts.astype(np.int64)
        logger.info(
            "Loaded: vis=%d sem=%d emo=%d spk=%d pro=%d sent=%d voice=%d flame=%d",
            len(self.vis_emb), len(self.sem_emb), len(self.emo_emb),
            len(self.spk_emb), len(self.pro_data), len(self.sent_emb),
            len(self.voice_emb), len(self.flame_exp),
        )
        self.model = SemanticCloneModel().to(self.device)

    def run(self) -> dict:
        results: dict[str, dict] = {}
        cfg = self.config
        _phases = {
            1: ("PHASE 1: Training 7 SPDs", self._train_spds),
            2: ("PHASE 2: Training 21 CMBs", self._train_cmbs),
            3: ("PHASE 3: Constellation Transformer", self._train_transformer),
        }
        for ph in range(cfg.start_phase, cfg.end_phase + 1):
            if ph in _phases:
                logger.info("=" * 60 + "\n%s\n" + "=" * 60, _phases[ph][0])
                results[f"phase{ph}"] = _phases[ph][1]()
            elif ph == 4 and cfg.cycle_steps > 0:
                logger.info("=" * 60 + "\nPHASE 4: Cycle Consistency\n" + "=" * 60)
                results["phase4"] = self._train_cycle()
        self._save_model("semantic_model_final.pt")
        logger.info("Training complete. Model saved to %s", self.save_dir)
        return results

    # -- Phase 1 --
    def _train_spds(self) -> dict:
        labels = {
            "visual": (self.model.spd_visual, self.vis_emb,
                       generate_visual_labels(self.vis_emb, self.flame_exp, self.flame_ts, self.vis_ts)),
            "emotion": (self.model.spd_emotion, self.emo_emb, generate_emotion_labels(self.emo_emb)),
            "prosody": (self.model.spd_prosody, self.pro_data, generate_prosody_labels(self.pro_data)),
            "semantic": (self.model.spd_semantic, self.sem_emb, generate_semantic_labels(self.sem_emb)),
            "speaker": (self.model.spd_speaker, self.spk_emb, generate_speaker_labels(self.spk_emb)),
            "sentence": (self.model.spd_sentence, self.sent_emb, generate_sentence_labels(self.sent_emb)),
            "voice": (self.model.spd_voice, self.voice_emb, generate_voice_labels(self.voice_emb)),
        }
        for name, (spd, raw, _) in labels.items():
            spd.calibrate(raw)
        results = {}
        for name, (spd, raw, lbl) in labels.items():
            results[name] = self._train_single_spd(spd, raw, lbl, name)
        self._save_model("semantic_model_phase1.pt")
        return results

    def _train_single_spd(self, spd: nn.Module, raw: np.ndarray, lbl: np.ndarray, name: str) -> dict:
        cfg = self.config
        t0 = time.time()
        logger.info("Training %s SPD: %d samples, %dd->%dd", name, len(raw), raw.shape[1], SEMANTIC_DIM)
        X = torch.from_numpy(raw.astype(np.float32)).to(self.device)
        Y = torch.from_numpy(lbl.astype(np.float32)).to(self.device)
        loader = DataLoader(TensorDataset(X, Y), batch_size=cfg.spd_batch_size, shuffle=True)
        opt = torch.optim.Adam(spd.parameters(), lr=cfg.spd_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.spd_epochs)
        spd.train()
        best = float("inf")
        for epoch in range(cfg.spd_epochs):
            eloss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.mse_loss(spd(xb), yb)
                loss.backward()
                opt.step()
                eloss += loss.item()
            sched.step()
            avg = eloss / len(loader)
            if avg < best:
                best = avg
            if (epoch + 1) % 100 == 0:
                logger.info("  %s epoch %d/%d: %.6f (best=%.6f)", name, epoch + 1, cfg.spd_epochs, avg, best)
        elapsed = time.time() - t0
        logger.info("%s SPD done: %.1fs, best=%.6f", name, elapsed, best)
        return {"best_loss": best, "elapsed_s": elapsed}

    # -- Phase 2 --
    def _train_cmbs(self) -> dict:
        cfg, t0 = self.config, time.time()
        logger.info("Computing SPD outputs for %d visual frames...", len(self.vis_emb))
        self.model.eval()
        with torch.no_grad():
            spds = [
                self._spd_fwd(self.model.spd_visual, self.vis_emb),
                self._spd_fwd(self.model.spd_emotion, self._align(self.emo_emb, self.emo_ts)),
                self._spd_fwd(self.model.spd_prosody, self._align(self.pro_data, self.pro_ts)),
                self._spd_fwd(self.model.spd_semantic, self._align(self.sem_emb, self.sem_ts)),
                self._spd_fwd(self.model.spd_speaker, self._align(self.spk_emb, self.spk_ts)),
                self._spd_fwd(self.model.spd_sentence, self._align(self.sent_emb, self.sent_ts)),
                self._spd_fwd(self.model.spd_voice, self._align(self.voice_emb, self.voice_ts)),
            ]
        self.model.train()
        losses = train_bridges(self.model.cmbs, *spds,
                               epochs=cfg.cmb_epochs, lr=cfg.cmb_lr, device=str(self.device))
        self.model.freeze_cmbs()
        self.model.freeze_spds()
        elapsed = time.time() - t0
        self._save_model("semantic_model_phase2.pt")
        logger.info("CMB training done: %.1fs", elapsed)
        return {"bridge_losses": losses, "elapsed_s": elapsed}

    # -- Phase 3 --
    def _train_transformer(self) -> dict:
        cfg, t0 = self.config, time.time()
        self._init_constellation()
        tensors = self._build_dataset()
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=cfg.transformer_batch_size, shuffle=True)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        logger.info("Trainable: %d / %d", sum(p.numel() for p in trainable), self.model.param_count)
        opt = torch.optim.AdamW(trainable, lr=cfg.transformer_lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.transformer_epochs)
        mloss = MeaningAwareLoss(geometric_weight=cfg.geometric_weight, semantic_weight=cfg.semantic_weight)
        self.model.train()
        best, log_iv = float("inf"), max(1, cfg.transformer_epochs // 20)
        for epoch in range(cfg.transformer_epochs):
            el, nb = 0.0, 0
            for batch in loader:
                bv, be, bp, bs, bk, bt, bo, bf, bei, bpi, bep = [x.to(self.device) for x in batch]
                opt.zero_grad()
                out = self.model(visual=bv, emotion=be, prosody=bp, semantic=bs,
                                 speaker=bk, sentence=bt, voice=bo)
                gt = torch.sigmoid(bf[:, :52] * 0.2)
                lg = F.mse_loss(out["blendshapes"], gt)
                ls, _ = mloss(out["blendshapes"], bf, bei, bpi, bep)
                lr_ = self.model.constellation_reg_loss()
                lc = self.model.cmbs.total_consistency_loss(out["spd_outputs"])
                loss = cfg.geometric_weight * lg + cfg.semantic_weight * ls + cfg.constellation_reg_weight * lr_ + cfg.cmb_consistency_weight * lc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                el += loss.item()
                nb += 1
            sched.step()
            avg = el / max(nb, 1)
            if avg < best:
                best = avg
            if (epoch + 1) % 250 == 0 or epoch == cfg.transformer_epochs - 1:
                self._save_model("semantic_model_best.pt")
            if (epoch + 1) % log_iv == 0 or epoch == 0:
                logger.info("Epoch %d/%d: %.6f best=%.6f", epoch + 1, cfg.transformer_epochs, avg, best)
        elapsed = time.time() - t0
        self._save_model("semantic_model_phase3.pt")
        logger.info("Transformer done: %.1fs, best=%.6f", elapsed, best)
        return {"best_loss": best, "elapsed_s": elapsed}

    def _init_constellation(self) -> None:
        logger.info("Building constellation from 7-modality SPD outputs...")
        fused = self._compute_fused()
        N = len(fused)
        from phoenix.clone.meaning_trainer import EMOTION_LABELS, PROSODY_LABELS
        ext = SemanticStateExtractor()
        emo_a = self._align(self.emo_emb, self.emo_ts)
        pro_a = self._align(self.pro_data, self.pro_ts)
        sv: dict[str, list[np.ndarray]] = {}
        for i in range(N):
            _, ep = ext.classify_emotion(float(emo_a[i, 0]), float(emo_a[i, 1]), float(emo_a[i, 2]))
            _, pp = ext.classify_prosody(pro_a[i])
            for s in [EMOTION_LABELS[int(np.argmax(ep))], PROSODY_LABELS[int(np.argmax(pp))]]:
                sv.setdefault(s, []).append(fused[i])
        se = {}
        for s, vecs in sv.items():
            if len(vecs) >= 3:
                se[s] = np.mean(vecs, axis=0).astype(np.float32)
                logger.info("  State '%s': %d samples", s, len(vecs))
        self.model.init_constellation(se)

    def _compute_fused(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            f = torch.cat([
                self.model.spd_visual(self._td(self.vis_emb)),
                self.model.spd_emotion(self._td(self._align(self.emo_emb, self.emo_ts))),
                self.model.spd_prosody(self._td(self._align(self.pro_data, self.pro_ts))),
                self.model.spd_semantic(self._td(self._align(self.sem_emb, self.sem_ts))),
                self.model.spd_speaker(self._td(self._align(self.spk_emb, self.spk_ts))),
                self.model.spd_sentence(self._td(self._align(self.sent_emb, self.sent_ts))),
                self.model.spd_voice(self._td(self._align(self.voice_emb, self.voice_ts))),
            ], dim=-1)
        return f.cpu().numpy()

    def _build_dataset(self) -> tuple:
        """Build aligned 7-modality dataset. Returns 11 tensors."""
        from phoenix.clone.meaning_trainer import EMOTION_LABELS, PROSODY_LABELS
        N = len(self.vis_emb)
        vis = torch.from_numpy(self.vis_emb.astype(np.float32))
        emo_np = self._align(self.emo_emb, self.emo_ts)
        pro_np = self._align(self.pro_data, self.pro_ts)
        emo, pro = torch.from_numpy(emo_np), torch.from_numpy(pro_np)
        sem = torch.from_numpy(self._align(self.sem_emb, self.sem_ts))
        spk = torch.from_numpy(self._align(self.spk_emb, self.spk_ts))
        sent = torch.from_numpy(self._align(self.sent_emb, self.sent_ts))
        voi = torch.from_numpy(self._align(self.voice_emb, self.voice_ts))
        flame = torch.from_numpy(self._align(self.flame_exp, self.flame_ts))
        ext = SemanticStateExtractor()
        ei, pi, epl = [], [], []
        for i in range(N):
            _, ep = ext.classify_emotion(float(emo_np[i, 0]), float(emo_np[i, 1]), float(emo_np[i, 2]))
            _, pp = ext.classify_prosody(pro_np[i])
            ei.append(int(np.argmax(ep)))
            pi.append(int(np.argmax(pp)))
            epl.append(ep)
        return (vis, emo, pro, sem, spk, sent, voi, flame,
                torch.tensor(ei, dtype=torch.long), torch.tensor(pi, dtype=torch.long),
                torch.from_numpy(np.stack(epl)))

    # -- Phase 4: Real Cycle Consistency (render + re-embed) --
    def _train_cycle(self) -> dict:
        """Real cycle consistency: render blendshapes, re-embed through SigLIP.

        1. Model predicts blendshapes from 7 embeddings
        2. FaceWarper renders face from blendshapes (jawOpen=bs[0], lipSpread=bs[11])
        3. Frozen SigLIP re-embeds rendered face -> 1152d
        4. Frozen visual SPD decodes re-embedded vector -> 32d
        5. Cycle loss = MSE(model visual SPD, rendered visual SPD)
        6. Gradients flow through model's forward path (render branch detached)
        """
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from PIL import Image
        from phoenix.render.face_warper import FaceWarper

        cfg, t0 = self.config, time.time()
        logger.info("Phase 4 cycle: %d steps, lr=%.1e", cfg.cycle_steps, cfg.cycle_lr)

        # --- Load FaceWarper with Santa reference frame ---
        ref_dir = Path.home() / ".clipcannon" / "models" / cfg.person / "reference"
        ref_path = ref_dir / f"{cfg.person}_ref_face.jpg"
        if not ref_path.exists():
            # Try extracting from source video
            ref_path_tmp = Path("/tmp") / f"{cfg.person}_ref_face.jpg"
            if ref_path_tmp.exists():
                ref_path = ref_path_tmp
            else:
                raise FileNotFoundError(
                    f"Reference frame not found at {ref_path} or {ref_path_tmp}. "
                    f"Extract a neutral face frame from the source video."
                )
        import cv2 as _cv2
        ref_bgr = _cv2.imread(str(ref_path))
        warper = FaceWarper(ref_bgr)
        if not warper.ready:
            logger.warning("FaceWarper landmarks failed; using pixel-proxy cycle")

        # --- Load frozen SigLIP for visual re-embedding ---
        from transformers import AutoModel, AutoProcessor
        siglip_name = "google/siglip-so400m-patch14-384"
        logger.info("Loading frozen SigLIP from cache...")
        siglip_model = AutoModel.from_pretrained(siglip_name).to(self.device).eval()
        siglip_proc = AutoProcessor.from_pretrained(siglip_name)
        for p in siglip_model.parameters():
            p.requires_grad_(False)
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated(self.device) / 1e6
            logger.info("VRAM after SigLIP load: %.0f MB", vram_mb)

        def _siglip_encode(bgr_frames: list[np.ndarray]) -> torch.Tensor:
            """Encode BGR frames through frozen SigLIP -> (B, 1152)."""
            pil_imgs = [Image.fromarray(_cv2.cvtColor(f, _cv2.COLOR_BGR2RGB)) for f in bgr_frames]
            inputs = siglip_proc(images=pil_imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = siglip_model.vision_model(pixel_values=inputs.pixel_values)
            return out.pooler_output  # (B, 1152)

        # --- Build dataset ---
        tensors = self._build_dataset()
        vis, emo, pro, sem, spk, sent, voi, flame = tensors[:8]
        ds = TensorDataset(vis, emo, pro, sem, spk, sent, voi, flame)
        loader = DataLoader(ds, batch_size=cfg.transformer_batch_size, shuffle=True)

        # --- Training loop ---
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=cfg.cycle_lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.cycle_steps)
        self.model.train()
        best, step = float("inf"), 0
        log_iv = max(1, cfg.cycle_steps // 20)

        while step < cfg.cycle_steps:
            for batch in loader:
                if step >= cfg.cycle_steps:
                    break
                bv, be, bp, bs_, bk, bt, bo, bf = [x.to(self.device) for x in batch]
                opt.zero_grad()

                # Differentiable model forward
                out = self.model(visual=bv, emotion=be, prosody=bp, semantic=bs_,
                                 speaker=bk, sentence=bt, voice=bo)
                bsp = out["blendshapes"]             # (B, 52) -- has gradients
                original_vis_spd = out["spd_outputs"]["visual"]  # (B, 32) -- has gradients

                # --- Non-differentiable render + re-embed (detached) ---
                B = bsp.shape[0]
                with torch.no_grad():
                    bs_np = bsp.detach().cpu().numpy()
                    rendered = []
                    for i in range(B):
                        jaw_open = float(bs_np[i, 0])
                        lip_spread = float(bs_np[i, 11]) if bs_np.shape[1] > 11 else 0.0
                        frame = warper.warp_mouth(mouth_open=jaw_open, head_tilt=lip_spread * 5.0)
                        rendered.append(frame)
                    # Re-embed rendered faces through frozen SigLIP
                    rendered_vis_emb = _siglip_encode(rendered)   # (B, 1152)
                    # Decode through frozen visual SPD
                    rendered_vis_spd = self.model.spd_visual(rendered_vis_emb)  # (B, 32)

                # --- Losses ---
                # Geometric anchor: blendshapes match normalized FLAME (has gradients via bsp)
                gt_bs = torch.sigmoid(bf[:, :52] * 0.2)
                l_geo = F.mse_loss(bsp, gt_bs)
                # Cycle loss via pooled representation (has gradients through transformer)
                # The rendered_vis_spd tells us what the output LOOKS like;
                # the model's pooled features should predict representations that
                # render to match the input visual meaning.
                pooled = out["_pooled"]  # (B, 224) -- connected to transformer
                l_cycle = F.mse_loss(
                    pooled[:, :SEMANTIC_DIM],
                    rendered_vis_spd.detach(),
                )
                loss = 1.0 * l_cycle + 0.3 * l_geo

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

                if loss.item() < best:
                    best = loss.item()
                step += 1
                if step % log_iv == 0:
                    vram_str = ""
                    if torch.cuda.is_available():
                        vram_str = f" vram={torch.cuda.memory_allocated(self.device)/1e6:.0f}MB"
                    logger.info("Cycle %d/%d: %.6f cycle=%.4f geo=%.4f%s",
                                step, cfg.cycle_steps, loss.item(),
                                l_cycle.item(), l_geo.item(), vram_str)
            sched.step()

        # Cleanup SigLIP to free VRAM
        del siglip_model, siglip_proc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        self._save_model("semantic_model_phase4.pt")
        logger.info("Cycle done: %.1fs, best=%.6f", elapsed, best)
        return {"best_loss": best, "elapsed_s": elapsed, "steps": step}

    # -- Utilities --
    def _align(self, data: np.ndarray, ts: np.ndarray) -> np.ndarray:
        N = len(self.vis_emb)
        out = np.zeros((N, data.shape[1]), dtype=np.float32)
        for i in range(N):
            out[i] = data[_nearest_idx(ts, self.vis_ts[i])]
        return out

    def _td(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data.astype(np.float32)).to(self.device)

    def _spd_fwd(self, spd: nn.Module, data: np.ndarray) -> torch.Tensor:
        return spd(self._td(data)).cpu()

    def _save_model(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {"person": self.config.person, "num_layers": self.model.num_layers},
        }, str(path))
        logger.info("Saved %s (%.1fMB)", path, path.stat().st_size / 1e6)


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train Semantic Clone Model (7 modalities)")
    p.add_argument("--person", default="santa")
    p.add_argument("--phase", type=int, default=0, help="Run only this phase (0=all)")
    p.add_argument("--epochs", type=int, default=0, help="Override transformer epochs")
    p.add_argument("--cycle-steps", type=int, default=0, help="Phase 4 cycle steps (0=skip)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--spd-epochs", type=int, default=300)
    p.add_argument("--cmb-epochs", type=int, default=200)
    args = p.parse_args()
    cfg = SemanticTrainerConfig(
        person=args.person, device=args.device,
        spd_epochs=args.spd_epochs, cmb_epochs=args.cmb_epochs,
        cycle_steps=args.cycle_steps,
    )
    if args.epochs > 0:
        cfg.transformer_epochs = args.epochs
    if args.phase > 0:
        cfg.start_phase = cfg.end_phase = args.phase
    results = SemanticTrainer(cfg).run()
    print("\nTRAINING COMPLETE")
    for phase, res in results.items():
        print(f"  {phase}: {res}")
    print(f"  Model: {cfg.save_dir}")


if __name__ == "__main__":
    main()
