"""Semantic Transformer Training Pipeline.

Four-phase training for the SemanticCloneModel:
  Phase 1: Train SPDs on Santa's data (pseudo-labels from ClipCannon) ~30 min
  Phase 2: Train CMBs on paired SPD outputs ~15 min
  Phase 3: Train constellation transformer ~2 hours
  Phase 4: Optional cycle consistency fine-tuning

CLI usage:
  python -m phoenix.clone.semantic_trainer --person santa
  python -m phoenix.clone.semantic_trainer --person santa --phase 3 --epochs 2000
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from phoenix.clone.cross_modal_bridges import CrossModalBridgeSet, train_bridges
from phoenix.clone.meaning_trainer import (
    MeaningAwareLoss,
    MeaningAlignedDataset,
    SemanticStateExtractor,
    _nearest_idx,
)
from phoenix.clone.semantic_decoders import (
    SEMANTIC_DIM,
    EmotionSPD,
    ProsodySPD,
    SemanticSPD,
    VisualSPD,
    generate_emotion_labels,
    generate_prosody_labels,
    generate_semantic_labels,
    generate_visual_labels,
)
from phoenix.clone.semantic_model import SemanticCloneModel

logger = logging.getLogger(__name__)


@dataclass
class SemanticTrainerConfig:
    """Configuration for the full training pipeline."""
    person: str = "santa"
    embeddings_path: str = ""
    flame_path: str = ""
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
    start_phase: int = 1
    end_phase: int = 3

    def __post_init__(self) -> None:
        base = Path.home() / ".clipcannon" / "models" / self.person
        if not self.embeddings_path:
            self.embeddings_path = str(base / "embeddings" / "all_embeddings.npz")
        if not self.flame_path:
            self.flame_path = str(base / "flame_params.npz")
        if not self.save_dir:
            self.save_dir = str(base / "semantic_model")


class SemanticTrainer:
    """Four-phase training pipeline for the Semantic Clone Model."""

    def __init__(self, config: SemanticTrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Load embeddings
        logger.info("Loading embeddings from %s", config.embeddings_path)
        self.emb_data = np.load(config.embeddings_path, allow_pickle=True)

        self.vis_emb = self.emb_data["vis_emb"]
        self.vis_ts = self.emb_data["vis_ts"]
        self.sem_emb = self.emb_data["sem_emb"]
        self.sem_ts = self.emb_data["sem_ts"]
        self.emo_data = self.emb_data["emo_data"]
        self.emo_ts = self.emb_data["emo_ts"]
        self.pro_data = self.emb_data["pro_data"]
        self.pro_ts = self.emb_data["pro_ts"]
        self.flame_exp = self.emb_data["flame_exp"]
        flame_ts_raw = self.emb_data["flame_ts"]
        self.flame_ts = (flame_ts_raw * 1000).astype(np.int64) if flame_ts_raw.max() < 10000 else flame_ts_raw.astype(np.int64)

        logger.info(
            "Data: vis=%d, sem=%d, emo=%d, pro=%d, flame=%d",
            len(self.vis_emb), len(self.sem_emb), len(self.emo_data),
            len(self.pro_data), len(self.flame_exp),
        )

        # The model
        self.model = SemanticCloneModel().to(self.device)

    def run(self) -> dict:
        """Run all configured training phases."""
        results = {}
        cfg = self.config

        if cfg.start_phase <= 1 <= cfg.end_phase:
            logger.info("=" * 60)
            logger.info("PHASE 1: Training Semantic Position Decoders")
            logger.info("=" * 60)
            results["phase1"] = self._train_spds()

        if cfg.start_phase <= 2 <= cfg.end_phase:
            logger.info("=" * 60)
            logger.info("PHASE 2: Training Cross-Modal Bridges")
            logger.info("=" * 60)
            results["phase2"] = self._train_cmbs()

        if cfg.start_phase <= 3 <= cfg.end_phase:
            logger.info("=" * 60)
            logger.info("PHASE 3: Training Constellation Transformer")
            logger.info("=" * 60)
            results["phase3"] = self._train_transformer()

        if cfg.start_phase <= 4 <= cfg.end_phase and cfg.cycle_steps > 0:
            logger.info("=" * 60)
            logger.info("PHASE 4: Cycle Consistency Fine-Tuning")
            logger.info("=" * 60)
            results["phase4"] = self._train_cycle()

        # Save final model
        self._save_model("semantic_model_final.pt")
        logger.info("Training complete. Model saved to %s", self.save_dir)
        return results

    def _train_spds(self) -> dict:
        """Train all four SPDs on pseudo-labels from ClipCannon analysis."""
        logger.info("Generating pseudo-labels...")
        vis_labels = generate_visual_labels(self.vis_emb, self.flame_exp, self.flame_ts, self.vis_ts)
        emo_labels = generate_emotion_labels(self.emo_data)
        pro_labels = generate_prosody_labels(self.pro_data)
        sem_labels = generate_semantic_labels(self.sem_emb)
        # Calibrate SPDs on per-person data ranges
        self.model.spd_visual.calibrate(self.vis_emb)
        self.model.spd_emotion.calibrate(self.emo_data)
        self.model.spd_prosody.calibrate(self.pro_data)
        self.model.spd_semantic.calibrate(self.sem_emb)
        results = {
            "visual": self._train_single_spd(self.model.spd_visual, self.vis_emb, vis_labels, "visual"),
            "emotion": self._train_single_spd(self.model.spd_emotion, self.emo_data, emo_labels, "emotion"),
            "prosody": self._train_single_spd(self.model.spd_prosody, self.pro_data, pro_labels, "prosody"),
            "semantic": self._train_single_spd(self.model.spd_semantic, self.sem_emb, sem_labels, "semantic"),
        }
        self._save_model("semantic_model_phase1.pt")
        return results

    def _train_single_spd(
        self,
        spd: nn.Module,
        raw_data: np.ndarray,
        labels: np.ndarray,
        name: str,
    ) -> dict:
        """Train a single SPD on its pseudo-labels."""
        cfg = self.config
        t0 = time.time()
        logger.info("Training %s SPD: %d samples, %d -> %d dims",
                     name, len(raw_data), raw_data.shape[1], SEMANTIC_DIM)

        X = torch.from_numpy(raw_data.astype(np.float32)).to(self.device)
        Y = torch.from_numpy(labels.astype(np.float32)).to(self.device)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=cfg.spd_batch_size, shuffle=True)

        optimizer = torch.optim.Adam(spd.parameters(), lr=cfg.spd_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.spd_epochs)

        spd.train()
        best_loss = float("inf")
        for epoch in range(cfg.spd_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = spd(x_batch)
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 100 == 0:
                logger.info("  %s SPD epoch %d/%d: loss=%.6f (best=%.6f)",
                            name, epoch + 1, cfg.spd_epochs, avg_loss, best_loss)

        elapsed = time.time() - t0
        logger.info("%s SPD done: %.1fs, best_loss=%.6f", name, elapsed, best_loss)
        return {"best_loss": best_loss, "elapsed_s": elapsed}

    # ------------------------------------------------------------------
    # Phase 2: Train CMBs
    # ------------------------------------------------------------------
    def _train_cmbs(self) -> dict:
        """Train cross-modal bridges on aligned SPD outputs."""
        cfg = self.config
        t0 = time.time()
        N = len(self.vis_emb)
        logger.info("Computing SPD outputs for %d visual frames...", N)

        self.model.eval()
        with torch.no_grad():
            spd_vis = self.model.spd_visual(torch.from_numpy(self.vis_emb.astype(np.float32)).to(self.device)).cpu()
            spd_emo = self.model.spd_emotion(torch.from_numpy(self._align_to_vis(self.emo_data, self.emo_ts)).to(self.device)).cpu()
            spd_pro = self.model.spd_prosody(torch.from_numpy(self._align_to_vis(self.pro_data, self.pro_ts)).to(self.device)).cpu()
            spd_sem = self.model.spd_semantic(torch.from_numpy(self._align_to_vis(self.sem_emb, self.sem_ts)).to(self.device)).cpu()

        self.model.train()
        losses = train_bridges(self.model.cmbs, spd_vis, spd_emo, spd_pro, spd_sem,
                               epochs=cfg.cmb_epochs, lr=cfg.cmb_lr, device=str(self.device))
        self.model.freeze_cmbs()
        self.model.freeze_spds()

        elapsed = time.time() - t0
        self._save_model("semantic_model_phase2.pt")
        logger.info("CMB training done: %.1fs", elapsed)
        return {"bridge_losses": losses, "elapsed_s": elapsed}

    # ------------------------------------------------------------------
    # Phase 3: Train Constellation Transformer
    # ------------------------------------------------------------------
    def _train_transformer(self) -> dict:
        """Train the constellation transformer on meaning-aware loss."""
        cfg = self.config
        t0 = time.time()

        # Build constellation from SPD outputs
        self._init_constellation()

        # Prepare aligned training data
        dataset = self._build_transformer_dataset()
        loader = DataLoader(
            dataset, batch_size=cfg.transformer_batch_size, shuffle=True,
        )

        # Optimizer: only train transformer + output heads (SPDs and CMBs frozen)
        trainable = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable.append(param)
        logger.info("Trainable params: %d / %d total",
                     sum(p.numel() for p in trainable),
                     self.model.param_count)

        optimizer = torch.optim.AdamW(trainable, lr=cfg.transformer_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.transformer_epochs)

        # Loss function
        meaning_loss = MeaningAwareLoss(
            geometric_weight=cfg.geometric_weight,
            semantic_weight=cfg.semantic_weight,
        )

        self.model.train()
        best_loss = float("inf")
        log_interval = max(1, cfg.transformer_epochs // 20)

        for epoch in range(cfg.transformer_epochs):
            epoch_loss = 0.0
            epoch_geo = 0.0
            epoch_reg = 0.0
            n_batches = 0

            for batch in loader:
                vis, emo, pro, sem, flame_gt, emo_idx, pro_idx, emo_probs = [
                    b.to(self.device) for b in batch
                ]

                optimizer.zero_grad()

                # Forward
                out = self.model(
                    visual=vis, emotion=emo, prosody=pro, semantic=sem,
                )

                # Geometric loss (blendshapes vs normalized FLAME)
                gt_normalized = torch.sigmoid(flame_gt[:, :52] * 0.2)
                loss_geo = F.mse_loss(out["blendshapes"], gt_normalized)

                # Meaning-aware semantic loss
                loss_sem, _ = meaning_loss(
                    out["blendshapes"], flame_gt,
                    emo_idx, pro_idx, emo_probs,
                )

                # Constellation regularization
                loss_reg = self.model.constellation_reg_loss()

                # CMB consistency (on frozen bridges)
                loss_cmb = self.model.cmbs.total_consistency_loss(out["spd_outputs"])

                # Total
                loss = (
                    cfg.geometric_weight * loss_geo
                    + cfg.semantic_weight * loss_sem
                    + cfg.constellation_reg_weight * loss_reg
                    + cfg.cmb_consistency_weight * loss_cmb
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_geo += loss_geo.item()
                epoch_reg += loss_reg.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save checkpoint every 250 epochs or at end, not every improvement
                if (epoch + 1) % 250 == 0 or epoch == cfg.transformer_epochs - 1:
                    self._save_model("semantic_model_best.pt")

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d: total=%.6f geo=%.6f reg=%.6f (best=%.6f)",
                    epoch + 1, cfg.transformer_epochs,
                    avg_loss, epoch_geo / max(n_batches, 1),
                    epoch_reg / max(n_batches, 1), best_loss,
                )

        elapsed = time.time() - t0
        self._save_model("semantic_model_phase3.pt")
        logger.info("Transformer training done: %.1fs, best_loss=%.6f", elapsed, best_loss)
        return {"best_loss": best_loss, "elapsed_s": elapsed}

    def _init_constellation(self) -> None:
        """Initialize constellation from SPD outputs clustered by state."""
        logger.info("Building constellation from SPD outputs...")
        fused_np = self._compute_fused_spd_outputs()
        N = len(fused_np)

        # Cluster into states using meaning labels
        extractor = SemanticStateExtractor()
        from phoenix.clone.meaning_trainer import EMOTION_LABELS, PROSODY_LABELS
        emo_aligned = self._align_to_vis(self.emo_data, self.emo_ts)
        pro_aligned = self._align_to_vis(self.pro_data, self.pro_ts)

        state_vectors: dict[str, list[np.ndarray]] = {}
        for i in range(N):
            _, ep = extractor.classify_emotion(
                float(emo_aligned[i, 0]), float(emo_aligned[i, 1]), float(emo_aligned[i, 2]),
            )
            _, pp = extractor.classify_prosody(pro_aligned[i])
            for state in [EMOTION_LABELS[int(np.argmax(ep))], PROSODY_LABELS[int(np.argmax(pp))]]:
                state_vectors.setdefault(state, []).append(fused_np[i])

        state_embeddings = {}
        for state, vecs in state_vectors.items():
            if len(vecs) >= 3:
                state_embeddings[state] = np.mean(vecs, axis=0).astype(np.float32)
                logger.info("  State '%s': %d samples", state, len(vecs))
        self.model.init_constellation(state_embeddings)

    def _compute_fused_spd_outputs(self) -> np.ndarray:
        """Compute fused SPD outputs for all visual frames."""
        self.model.eval()
        with torch.no_grad():
            vis = torch.from_numpy(self.vis_emb.astype(np.float32)).to(self.device)
            emo = torch.from_numpy(self._align_to_vis(self.emo_data, self.emo_ts)).to(self.device)
            pro = torch.from_numpy(self._align_to_vis(self.pro_data, self.pro_ts)).to(self.device)
            sem = torch.from_numpy(self._align_to_vis(self.sem_emb, self.sem_ts)).to(self.device)
            fused = torch.cat([
                self.model.spd_visual(vis), self.model.spd_emotion(emo),
                self.model.spd_prosody(pro), self.model.spd_semantic(sem),
            ], dim=-1)
        return fused.cpu().numpy()

    def _align_to_vis(self, data: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Align data to visual timestamps using nearest neighbor."""
        N = len(self.vis_emb)
        aligned = np.zeros((N, data.shape[1]), dtype=np.float32)
        for i in range(N):
            idx = _nearest_idx(ts, self.vis_ts[i])
            aligned[i] = data[idx]
        return aligned

    def _build_transformer_dataset(self) -> TensorDataset:
        """Build aligned dataset for transformer training."""
        from phoenix.clone.meaning_trainer import EMOTION_LABELS, PROSODY_LABELS
        N = len(self.vis_emb)

        vis = torch.from_numpy(self.vis_emb.astype(np.float32))
        emo_np = self._align_to_vis(self.emo_data, self.emo_ts)
        pro_np = self._align_to_vis(self.pro_data, self.pro_ts)
        emo = torch.from_numpy(emo_np)
        pro = torch.from_numpy(pro_np)
        sem = torch.from_numpy(self._align_to_vis(self.sem_emb, self.sem_ts))
        flame = torch.from_numpy(self._align_to_vis(self.flame_exp, self.flame_ts))

        extractor = SemanticStateExtractor()
        emo_indices, pro_indices, emo_probs_list = [], [], []
        for i in range(N):
            _, ep = extractor.classify_emotion(float(emo_np[i, 0]), float(emo_np[i, 1]), float(emo_np[i, 2]))
            _, pp = extractor.classify_prosody(pro_np[i])
            emo_indices.append(int(np.argmax(ep)))
            pro_indices.append(int(np.argmax(pp)))
            emo_probs_list.append(ep)

        return TensorDataset(
            vis, emo, pro, sem, flame,
            torch.tensor(emo_indices, dtype=torch.long),
            torch.tensor(pro_indices, dtype=torch.long),
            torch.from_numpy(np.stack(emo_probs_list)),
        )

    # ------------------------------------------------------------------
    # Phase 4: Cycle consistency (optional)
    # ------------------------------------------------------------------
    def _train_cycle(self) -> dict:
        """Optional cycle consistency fine-tuning (placeholder)."""
        logger.info("Phase 4 cycle consistency not yet implemented. Skipping.")
        return {"status": "skipped"}

    # ------------------------------------------------------------------
    # Save/Load
    # ------------------------------------------------------------------
    def _save_model(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "person": self.config.person,
                "num_layers": self.model.num_layers,
            },
        }, str(path))
        logger.info("Saved %s (%.1fMB)", path, path.stat().st_size / 1e6)


def main() -> None:
    """CLI entry point: python -m phoenix.clone.semantic_trainer --person santa."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train Semantic Clone Model")
    p.add_argument("--person", default="santa")
    p.add_argument("--phase", type=int, default=0, help="Run only this phase (0=all)")
    p.add_argument("--epochs", type=int, default=0, help="Override transformer epochs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--spd-epochs", type=int, default=300)
    p.add_argument("--cmb-epochs", type=int, default=200)
    args = p.parse_args()
    cfg = SemanticTrainerConfig(person=args.person, device=args.device,
                                spd_epochs=args.spd_epochs, cmb_epochs=args.cmb_epochs)
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
