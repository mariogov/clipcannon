"""Gaussian avatar trainer with L1 + SSIM + LPIPS perceptual loss.

Trains GaussianAvatarModel from FLAME params + video face crops.
200K iters, tex_size=256 (58K Gaussians), 512x512 crops, FP16 AMP.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _ssim_loss(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
) -> torch.Tensor:
    """SSIM loss (1 - SSIM) between (B,C,H,W) image batches."""
    C = img1.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return 1 - ssim_map.mean()


def extract_face_crops_512(
    video_path: str | Path, flame_params_path: str | Path,
    output_path: str | Path, crop_size: int = 512,
) -> Path:
    """Re-extract face crops at crop_size from source video using insightface."""
    from insightface.app import FaceAnalysis

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = np.load(str(flame_params_path))
    timestamps = params["timestamps"]
    n_frames = len(timestamps)
    logger.info("Extracting %d face crops at %dx%d from %s",
                n_frames, crop_size, crop_size, video_path.name)

    face_app = FaceAnalysis(providers=["CUDAExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    crops = np.zeros((n_frames, crop_size, crop_size, 3), dtype=np.uint8)
    last_bbox = None
    extracted = 0

    for i, ts in enumerate(timestamps):
        frame_num = int(round(ts * video_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            if last_bbox is not None:
                crops[i] = crops[max(0, i - 1)]
            continue

        faces = face_app.get(frame)
        if len(faces) > 0:
            bbox = faces[0].bbox.astype(int)
            last_bbox = bbox
        elif last_bbox is not None:
            bbox = last_bbox
        else:
            continue

        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        half = int(max(bw, bh) * 0.65)
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(frame.shape[1], cx + half)
        y2 = min(frame.shape[0], cy + half)
        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crops[i] = cv2.resize(crop, (crop_size, crop_size))
            extracted += 1

        if (i + 1) % 500 == 0:
            logger.info("  Extracted %d/%d crops...", i + 1, n_frames)

    cap.release()
    np.savez_compressed(str(output_path), crops=crops)
    logger.info("Saved %d face crops (%d valid) to %s",
                n_frames, extracted, output_path)
    return output_path


class GaussianTrainer:
    """Trains Gaussian avatar with L1 + SSIM + LPIPS loss, warmup + cosine LR."""

    def __init__(
        self,
        flame_params_path: str | Path,
        face_crops_path: str | Path | None = None,
        output_path: str | Path | None = None,
        num_iters: int = 200_000,
        batch_size: int = 4,
        lr: float = 5e-4,
        lr_min: float = 1e-5,
        warmup_iters: int = 1000,
        tex_size: int = 256,
        render_size: int = 512,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        lpips_weight: float = 0.1,
        checkpoint_every: int = 50_000,
        log_every: int = 1000,
        device_id: int = 0,
    ) -> None:
        self.flame_params_path = Path(flame_params_path)
        if face_crops_path is None:
            self.face_crops_path = self.flame_params_path.parent / "face_crops_512.npz"
        else:
            self.face_crops_path = Path(face_crops_path)
        if output_path is None:
            self.output_path = self.flame_params_path.parent / "gaussian_avatar_v2.pt"
        else:
            self.output_path = Path(output_path)

        self.num_iters = num_iters
        self.batch_size = batch_size
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_iters = warmup_iters
        self.tex_size = tex_size
        self.render_size = render_size
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.device = torch.device(f"cuda:{device_id}")

        self._flame = None
        self._avatar = None
        self._lpips_model = None
        self._expressions = None
        self._jaw_poses = None
        self._target_images = None

    def _load_data(self) -> None:
        """Load FLAME params and face crop images."""
        params = np.load(str(self.flame_params_path))
        self._expressions = torch.from_numpy(params["expression"]).to(
            self.device, torch.float32,
        )
        self._jaw_poses = torch.from_numpy(params["jaw_pose"]).to(
            self.device, torch.float32,
        )
        logger.info(
            "Loaded FLAME params: %d frames, expr %s, jaw %s",
            len(self._expressions),
            tuple(self._expressions.shape),
            tuple(self._jaw_poses.shape),
        )

        crops_data = np.load(str(self.face_crops_path))
        crops = crops_data["crops"]  # (N, H, W, 3) uint8
        logger.info("Loaded %d face crops: %s", len(crops), crops.shape)

        # Resize to render_size and convert to float [0, 1]
        resized = []
        for crop in crops:
            r = cv2.resize(crop, (self.render_size, self.render_size))
            resized.append(r)
        images = np.array(resized, dtype=np.float32) / 255.0
        # BGR to RGB
        images = images[:, :, :, ::-1].copy()
        self._target_images = torch.from_numpy(images).to(self.device)
        logger.info("Target images: %s", tuple(self._target_images.shape))

    def _init_models(self) -> None:
        """Initialize FLAME, Gaussian avatar, and LPIPS models."""
        from phoenix.render.flame_model import FlameModel, FlameModelConfig
        from phoenix.render.gsplat_avatar import AvatarRenderConfig, GaussianAvatarModel

        flame_config = FlameModelConfig()
        self._flame = FlameModel(flame_config).to(self.device)
        self._flame.eval()
        for p in self._flame.parameters():
            p.requires_grad_(False)

        avatar_config = AvatarRenderConfig(
            width=self.render_size,
            height=self.render_size,
            tex_size=self.tex_size,
            num_expression_in=100,
            num_basis=20,
            fov_y=25.0,
            bg_color=(0.0, 0.0, 0.0),
        )
        self._avatar = GaussianAvatarModel(self._flame, avatar_config)
        self._avatar = self._avatar.to(self.device)
        num_gs = self._avatar.bind_to_mesh()
        logger.info(
            "Gaussian avatar: %d Gaussians (tex_size=%d)", num_gs, self.tex_size,
        )

        # LPIPS perceptual loss (frozen VGG backbone)
        import lpips
        self._lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        self._lpips_model.eval()
        for p in self._lpips_model.parameters():
            p.requires_grad_(False)
        logger.info("LPIPS perceptual loss model loaded")

    def _get_lr(self, step: int) -> float:
        """Linear warmup for warmup_iters, then cosine decay to lr_min."""
        if step <= self.warmup_iters:
            return self.lr * step / self.warmup_iters
        progress = (step - self.warmup_iters) / max(
            1, self.num_iters - self.warmup_iters,
        )
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (
            1 + math.cos(math.pi * progress)
        )

    def _sample_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of training frames."""
        N = len(self._expressions)
        indices = torch.randint(0, N, (self.batch_size,), device=self.device)
        return (
            self._expressions[indices],
            self._jaw_poses[indices],
            self._target_images[indices],
        )

    def train(self) -> None:
        """Run the full training loop with L1 + SSIM + LPIPS loss."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading training data...")
        self._load_data()

        logger.info("Initializing models...")
        self._init_models()

        shape_params = torch.zeros(1, 300, device=self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam(self._avatar.parameters(), lr=self.lr)

        logger.info(
            "Training config: %d iters, batch=%d, lr=%.1e->%.1e, "
            "tex=%d, render=%d, loss=%.1f*L1+%.1f*SSIM+%.1f*LPIPS",
            self.num_iters, self.batch_size, self.lr, self.lr_min,
            self.tex_size, self.render_size,
            self.l1_weight, self.ssim_weight, self.lpips_weight,
        )

        train_start = time.time()
        best_loss = float("inf")
        running_l1 = 0.0
        running_ssim = 0.0
        running_lpips = 0.0
        running_total = 0.0
        running_count = 0

        for step in range(1, self.num_iters + 1):
            # Update LR with warmup + cosine schedule
            lr_now = self._get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)

            expr, jaw, target = self._sample_batch()
            B = expr.shape[0]

            # Forward pass (no AMP on gsplat -- it needs fp32 internally)
            batch_shape = shape_params.expand(B, -1)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                verts = self._flame(batch_shape, expr, jaw_pose=jaw)
            result = self._avatar.render(
                verts.float(), expression_weights=expr,
            )
            rendered = result["image"]  # (B, H, W, 3)
            rendered_bchw = rendered.permute(0, 3, 1, 2)
            target_bchw = target.permute(0, 3, 1, 2)

            loss_l1 = F.l1_loss(rendered_bchw, target_bchw)
            loss_ssim = _ssim_loss(rendered_bchw, target_bchw)

            # LPIPS: scale to [-1,1], keep grad graph connected
            rendered_lpips = (rendered_bchw * 2 - 1).clamp(-1, 1)
            with torch.no_grad():
                target_lpips = (target_bchw * 2 - 1).clamp(-1, 1)
            loss_lpips = self._lpips_model(rendered_lpips, target_lpips).mean()

            loss = (
                self.l1_weight * loss_l1
                + self.ssim_weight * loss_ssim
                + self.lpips_weight * loss_lpips
            )

            loss.backward()
            optimizer.step()

            # Track running averages
            l1_val = loss_l1.item()
            ssim_val = loss_ssim.item()
            lpips_val = loss_lpips.item()
            total_val = loss.item()
            running_l1 += l1_val
            running_ssim += ssim_val
            running_lpips += lpips_val
            running_total += total_val
            running_count += 1

            if total_val < best_loss:
                best_loss = total_val

            # Logging
            if step % self.log_every == 0 or step == 1:
                elapsed = time.time() - train_start
                it_per_sec = step / max(elapsed, 0.01)
                remaining = (self.num_iters - step) / max(it_per_sec, 0.01)
                avg_l1 = running_l1 / running_count
                avg_ssim = running_ssim / running_count
                avg_lpips = running_lpips / running_count
                avg_total = running_total / running_count
                logger.info(
                    "Step %d/%d | L1=%.4f SSIM=%.4f LPIPS=%.4f Total=%.4f | "
                    "Best=%.4f | LR=%.2e | %.1f it/s | ~%.0fm left",
                    step, self.num_iters,
                    avg_l1, avg_ssim, avg_lpips, avg_total,
                    best_loss, lr_now, it_per_sec, remaining / 60,
                )
                running_l1 = running_ssim = running_lpips = 0.0
                running_total = 0.0
                running_count = 0

            # Checkpoints
            if step % self.checkpoint_every == 0:
                self._save_checkpoint(step)

        total_time = time.time() - train_start
        logger.info(
            "Training complete: %d iters in %.1fm (%.1f it/s), best_loss=%.4f",
            self.num_iters, total_time / 60, self.num_iters / total_time, best_loss,
        )
        self._save_checkpoint(self.num_iters, final=True)

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "avatar_state_dict": self._avatar.state_dict(),
            "config": self._avatar.config,
            "tex_size": self.tex_size,
            "render_size": self.render_size,
            "num_gaussians": self._avatar.num_gaussians,
        }
        if final:
            path = self.output_path
        else:
            path = self.output_path.parent / f"gaussian_avatar_v2_step{step}.pt"
        torch.save(checkpoint, str(path))
        logger.info("Saved checkpoint to %s (step %d)", path, step)

    def render_test_frame(self, output_path: str | Path) -> None:
        """Render a test frame from the middle of the training data."""
        output_path = Path(output_path)
        mid = len(self._expressions) // 2
        expr = self._expressions[mid:mid + 1]
        jaw = self._jaw_poses[mid:mid + 1]
        shape = torch.zeros(1, 300, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            verts = self._flame(shape, expr, jaw_pose=jaw)
            result = self._avatar.render(verts, expression_weights=expr)
            img = result["image"][0].cpu().numpy()

        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_bgr)
        logger.info("Test frame saved to %s", output_path)


def main() -> None:
    """CLI entry point for Gaussian avatar training."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train Gaussian avatar v2")
    d = Path.home() / ".clipcannon/models/santa"
    p.add_argument("--flame-params", default=str(d / "flame_params.npz"))
    p.add_argument("--face-crops", default=None)
    p.add_argument("--output", default=str(d / "gaussian_avatar_v2.pt"))
    p.add_argument("--video", default=None, help="Source video for 512x512 crop extraction")
    p.add_argument("--num-iters", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=1000)
    p.add_argument("--tex-size", type=int, default=256)
    p.add_argument("--render-size", type=int, default=512)
    p.add_argument("--l1-weight", type=float, default=0.8)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--test-frame", default="/tmp/gaussian_v2_test.jpg")
    a = p.parse_args()

    face_crops = a.face_crops or str(Path(a.flame_params).parent / "face_crops_512.npz")
    if a.video and not Path(face_crops).exists():
        extract_face_crops_512(a.video, a.flame_params, face_crops, crop_size=512)

    trainer = GaussianTrainer(
        flame_params_path=a.flame_params, face_crops_path=face_crops,
        output_path=a.output, num_iters=a.num_iters, batch_size=a.batch_size,
        lr=a.lr, lr_min=a.lr_min, warmup_iters=a.warmup_iters,
        tex_size=a.tex_size, render_size=a.render_size,
        l1_weight=a.l1_weight, ssim_weight=a.ssim_weight,
        lpips_weight=a.lpips_weight, checkpoint_every=a.checkpoint_every,
    )
    trainer.train()
    if a.test_frame:
        trainer.render_test_frame(a.test_frame)


if __name__ == "__main__":
    main()
